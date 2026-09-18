from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from grasp.target_tracking import PoseHistory, TargetTracker, wait_stable_target, tracked_approach


def pose(x=0., angle=0.):
    result = np.eye(4)
    result[0, 3] = x
    result[:3, :3] = Rotation.from_euler('z', angle, degrees=True).as_matrix()
    return result


def test_pose_history_interpolates_camera_time_and_refuses_extrapolation():
    history = PoseHistory()
    history.add(pose(0, 0), [10.]*6)
    history.add(pose(.1, 90), [10.08]*6)
    np.testing.assert_allclose(history.at(10.04), pose(.05, 45), atol=1e-12)
    with pytest.raises(ValueError, match='外推'):
        history.at(10.1)
    with pytest.raises(ValueError, match='倒退'):
        history.add(pose(), [9.]*6)


def test_pose_history_rejects_feedback_gaps_and_split_frames():
    history = PoseHistory()
    with pytest.raises(ValueError, match='50 ms'):
        history.add(pose(), [10., 10.1])
    history.add(pose(), [10.]*6)
    history.add(pose(), [10.2]*6)
    with pytest.raises(ValueError, match='100 ms'):
        history.at(10.1)


def tracker_fixture(monkeypatch):
    history = PoseHistory()
    for t in np.arange(10., 11.01, .04):
        history.add(pose((t-10)*.1, (t-10)*30), [t]*6)
    preview = SimpleNamespace(value=None)
    preview.sample = lambda: preview.value
    clock = [10.5]
    tracker = TargetTracker(preview, history, np.eye(4), {}, None, now=lambda: clock[0])
    monkeypatch.setattr('grasp.target_tracking.target_point',
                        lambda sample, *args: sample[1].point)
    tracker.set_target([.2, .1, .3])

    def publish(t, base=(.2, .1, .3), *, valid=True, domain='timestamp_domain.global_time'):
        camera = np.linalg.inv(history.at(t)) @ np.r_[base, 1]
        frame = SimpleNamespace(metadata=dict(timestamp_domain=domain, color_timestamp_ms=t*1000),
                                point=dict(valid=valid, xyz_camera_m=camera[:3], reason='lost'))
        preview.value = (t, frame, None, None)
        return frame
    return tracker, publish, clock


def test_camera_motion_does_not_trigger_target_movement(monkeypatch):
    tracker, publish, clock = tracker_fixture(monkeypatch)
    for t in (10.05, 10.1, 10.2, 10.3):
        publish(t)
        assert tracker.monitor() is None
    # Inference completes much later than exposure; use historical flange pose.
    publish(10.4)
    clock[0] = 11.
    observation, reason = tracker.observation()
    assert reason is None
    np.testing.assert_allclose(observation['base'], [.2, .1, .3], atol=1e-12)


def test_two_distinct_frames_confirm_motion_and_duplicates_do_not(monkeypatch):
    tracker, publish, _ = tracker_fixture(monkeypatch)
    publish(10.05, (.23, .1, .3))
    assert tracker.monitor() is None
    assert tracker.monitor() is None
    publish(10.1, (.23, .1, .3))
    assert '连续两帧' in tracker.monitor()


def test_jitter_resets_motion_confirmation(monkeypatch):
    tracker, publish, _ = tracker_fixture(monkeypatch)
    publish(10.05, (.22, .1, .3)); assert tracker.monitor() is None
    publish(10.1, (.203, .1, .3)); assert tracker.monitor() is None
    publish(10.15, (.22, .1, .3)); assert tracker.monitor() is None


@pytest.mark.parametrize('fault', ['lost', 'hardware_time', 'stale', 'future', 'pose_gap'])
def test_invalid_vision_requests_stop(monkeypatch, fault):
    tracker, publish, clock = tracker_fixture(monkeypatch)
    frame = publish(10.1)
    if fault == 'lost': frame.point['valid'] = False
    if fault == 'hardware_time': frame.metadata['timestamp_domain'] = 'timestamp_domain.hardware_clock'
    if fault == 'stale': clock[0] = 12.
    if fault == 'future': clock[0] = 9.
    if fault == 'pose_gap': tracker.history = PoseHistory()
    assert tracker.monitor()


def test_red_block_count_needs_three_distinct_frames(monkeypatch):
    tracker, publish, _ = tracker_fixture(monkeypatch)
    for t, count in ((10.05, 0), (10.1, 2)):
        frame = publish(t, valid=False)
        frame.point.update(reason='red_block_count', count=count)
        assert tracker.monitor() is None
        # Re-polling the same inference result must not count as another frame.
        assert tracker.monitor() is None
    frame = publish(10.15, valid=False)
    frame.point.update(reason='red_block_count', count=0)
    reason = tracker.monitor()
    assert reason == 'red_block_count连续3帧（各帧数量：0,2,0）'


def test_valid_frame_resets_red_block_count_confirmation(monkeypatch):
    tracker, publish, _ = tracker_fixture(monkeypatch)
    frame = publish(10.05, valid=False)
    frame.point.update(reason='red_block_count', count=0)
    assert tracker.monitor() is None
    publish(10.1)
    assert tracker.monitor() is None
    for t in (10.15, 10.2):
        frame = publish(t, valid=False)
        frame.point.update(reason='red_block_count', count=0)
        assert tracker.monitor() is None


def test_duplicate_eventually_expires(monkeypatch):
    tracker, publish, clock = tracker_fixture(monkeypatch)
    publish(10.1)
    assert tracker.monitor() is None
    clock[0] = 12.
    assert '过期' in tracker.monitor()


def test_reacquisition_needs_post_stop_stable_frames():
    clock = [10.]
    frames = iter([dict(stamp=9., base=np.zeros(3)),
                   dict(stamp=10.01, base=np.zeros(3)),
                   dict(stamp=10.2, base=np.ones(3)),  # moving: reset
                   dict(stamp=10.4, base=np.ones(3)),
                   dict(stamp=10.6, base=np.ones(3))])
    tracker = SimpleNamespace(observation=lambda: (next(frames), None))
    events = []
    result = wait_stable_target(tracker, lambda: None, events.append,
                               now=lambda: clock[0], wall=lambda: 10.,
                               sleep=lambda dt: clock.__setitem__(0, clock[0]+dt))
    assert result['stamp'] == 10.6
    assert events[-1]['event'] == 'tracking_target_stable'


def test_lost_target_timeout_never_returns_a_target():
    clock = [0.]
    tracker = SimpleNamespace(observation=lambda: (None, 'red_block_count'))
    with pytest.raises(TimeoutError, match='red_block_count'):
        wait_stable_target(tracker, lambda: None, lambda row: None, timeout=.03,
                           now=lambda: clock[0], wall=lambda: 0.,
                           sleep=lambda dt: clock.__setitem__(0, clock[0]+dt))


def test_approach_replans_after_hold_then_verifies_before_grasp(monkeypatch):
    import grasp.target_tracking as module
    a, b, c = np.zeros(3), np.ones(3)*.02, np.ones(3)*.04
    # First motion stops. Target moves again during the first planning pass.
    observations = iter([a, b, c, c, c])
    calls = []
    monkeypatch.setattr(module, 'wait_stable_target',
                        lambda *args: dict(base=next(observations)))
    outcomes = iter([False, True])
    def execute(*args, **kwargs):
        calls.append('motion')
        return next(outcomes)
    monkeypatch.setattr(module, 'execute_path', execute)
    tracker = SimpleNamespace(preview=SimpleNamespace(), threshold_m=.01,
                              set_target=lambda target: None, monitor=lambda: None)
    def build(observation):
        calls.append(observation['base'].copy())
        return dict(red_block_base_xyz_m=observation['base']), ([], [])
    result = tracked_approach(None, tracker, dict(red_block_base_xyz_m=a), ([], []),
                              build, lambda: None, lambda row: None)
    assert calls[0] == calls[-1] == 'motion'
    np.testing.assert_equal(calls[1], b)
    np.testing.assert_equal(calls[2], c)
    np.testing.assert_equal(result['red_block_base_xyz_m'], c)


def test_replan_limit_prevents_another_motion(monkeypatch):
    import grasp.target_tracking as module
    n = [0]
    def observe(*args):
        n[0] += 1
        return dict(base=np.ones(3)*n[0])
    monkeypatch.setattr(module, 'wait_stable_target', observe)
    monkeypatch.setattr(module, 'execute_path', lambda *args, **kw: pytest.fail('must not move'))
    tracker = SimpleNamespace(preview=SimpleNamespace(), threshold_m=.01)
    with pytest.raises(RuntimeError, match='次数'):
        tracked_approach(None, tracker, dict(red_block_base_xyz_m=np.zeros(3)), ([], []),
                         lambda obs: (dict(red_block_base_xyz_m=obs['base']), ([], [])),
                         lambda: None, lambda row: None, max_replans=1)


def test_near_lock_skips_post_arrival_visual_reacquisition(monkeypatch):
    import grasp.target_tracking as module
    calls = []
    monkeypatch.setattr(module, 'wait_stable_target',
                        lambda *args: dict(base=np.zeros(3)))
    def execute(*args, **kwargs):
        calls.append(kwargs['monitor'])
        return True
    monkeypatch.setattr(module, 'execute_path', execute)
    monitor = lambda: None
    tracker = SimpleNamespace(preview=SimpleNamespace(), threshold_m=.05,
                              set_target=lambda target: None, monitor=lambda: 'unused')
    events = []
    report = dict(red_block_base_xyz_m=np.zeros(3))
    result = tracked_approach(None, tracker, report, ([], []),
                              lambda obs: pytest.fail('must not replan'),
                              lambda: None, events.append, monitor=monitor,
                              verify_after_reached=False)
    assert result is report
    assert calls == [monitor]
    assert events[-1]['event'] == 'tracking_pregrasp_locked'
