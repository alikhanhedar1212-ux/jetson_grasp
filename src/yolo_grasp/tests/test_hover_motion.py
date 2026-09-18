from types import SimpleNamespace
import numpy as np
import pytest
from grasp.hover_motion import timed_path, execute_path


def test_time_scaling_speed_and_endpoints():
    path = np.zeros((301, 6))
    path[:, 0] = np.deg2rad(np.linspace(0, 60, 301))
    path[:, 1] = np.deg2rad(np.sin(np.linspace(0, np.pi, 301))*10)
    times, targets = timed_path(path)
    np.testing.assert_allclose(targets[[0, -1]], path[[0, -1]], atol=1e-14)
    speeds = np.max(abs(np.diff(targets, axis=0)), axis=1)/np.diff(times)
    assert speeds.max() <= np.deg2rad(2.001)
    assert speeds[0] < np.deg2rad(.001) and speeds[-1] < np.deg2rad(.001)
    assert np.diff(times).max() <= .020001
    assert targets[:, 1].max() > np.deg2rad(9.99)


class FakeController:
    def __init__(self):
        self.clock = 10.
        self.joints = np.zeros(6)
        self.locked = False
        self.active = None
        self.prepared = True
        self.timeout = 1.
        self.auto = False
        self.calls = []
        self.stops = []
        self.mode = 1
        self.follow = True
        self.fresh = True
        self.arm = SimpleNamespace(
            _msg_mode=SimpleNamespace(move_spd_rate_ctrl=99),
            get_auto_set_motion_mode_enabled=lambda: self.auto,
            set_auto_set_motion_mode_enabled=self.set_auto,
            move_j=self.move)
    def set_auto(self, value): self.auto = value
    def move(self, q):
        self.calls.append((self.clock, np.array(q)))
        if self.follow: self.joints = np.array(q)
    def state(self):
        return SimpleNamespace(joints=self.joints.copy(), healthy=lambda: None,
            enabled=[True]*6, ctrl_mode=1, mode_feedback=self.mode,
            stamps=np.full(6, self.clock if self.fresh else 0.), motion_status=0)
    def validate_joints(self, q):
        assert np.isfinite(q).all()
    def stop(self, reason): self.locked = True; self.stops.append(reason)
    def sleep(self, seconds): self.clock += seconds
    def tick(self):
        if getattr(self, 'guard_goal', None) is not None:
            tolerance = np.deg2rad(getattr(self, 'guard_tolerance_deg', .3))
            assert np.all(self.joints >= self.guard_start-tolerance)
            assert np.all(self.joints <= self.guard_goal+tolerance)


def run(c, tick=None):
    path = np.zeros((11, 6)); path[:, 0] = np.deg2rad(np.linspace(0, 3, 11))
    times, targets = timed_path(path)
    events = []
    execute_path(c, times, targets, tick or c.tick, events.append,
                 now=lambda: c.clock, wall=lambda: c.clock, sleep=c.sleep)
    return times, targets, events


def test_visual_interrupt_holds_and_settles_without_emergency_stop():
    c = FakeController()
    path = np.zeros((11, 6)); path[:, 0] = np.deg2rad(np.linspace(0, 3, 11))
    times, targets = timed_path(path)
    events = []
    reached = execute_path(c, times, targets, c.tick, events.append,
                           now=lambda: c.clock, wall=lambda: c.clock, sleep=c.sleep,
                           monitor=lambda: 'target moved' if len(c.calls) >= 10 else None)
    assert reached is False
    assert len(c.calls) == 11  # ten old targets, exactly one hold command
    np.testing.assert_equal(c.calls[-1][1], c.calls[-2][1])
    assert c.clock-c.calls[-1][0] >= .5
    assert not c.stops and not c.locked and not c.auto
    assert events[-1]['event'] == 'tracking_hold_settled'


def test_visual_hold_failure_uses_emergency_stop():
    c = FakeController()
    path = np.zeros((11, 6)); path[:, 0] = np.deg2rad(np.linspace(0, 3, 11))
    times, targets = timed_path(path)
    def tick():
        if len(c.calls) >= 11:
            raise RuntimeError('feedback lost during hold')
    with pytest.raises(RuntimeError, match='feedback lost'):
        execute_path(c, times, targets, tick, lambda row: None,
                     now=lambda: c.clock, wall=lambda: c.clock, sleep=c.sleep,
                     monitor=lambda: 'moved' if len(c.calls) >= 10 else None)
    assert c.locked and c.stops and not c.auto


def test_continuous_updates_only_dwell_at_endpoint():
    c = FakeController()
    times, targets, events = run(c)
    assert len(c.calls) == len(times)
    assert np.diff([t for t, _ in c.calls]).max() < .021
    np.testing.assert_allclose(c.joints, targets[-1])
    assert c.clock-c.calls[-1][0] >= .5
    assert events[-1]['event'] == 'continuous_reached'
    assert not c.auto and not c.stops
    assert c.arm._msg_mode.move_spd_rate_ctrl == 3


def test_parked_offset_below_hold_tolerance_still_starts():
    """A snap while parked must not block the next continuous motion."""
    c = FakeController(); c.joints = np.deg2rad([.5]*6)
    times, targets, events = run(c)
    assert events[-1]['event'] == 'continuous_reached' and not c.stops


def test_parked_offset_beyond_hold_tolerance_is_rejected():
    c = FakeController(); c.joints = np.deg2rad([.9]*6)
    with pytest.raises(RuntimeError, match='起点与反馈'):
        run(c)
    assert c.locked


@pytest.mark.parametrize('fault', ['stop', 'mode', 'stale', 'scheduler'])
def test_fault_stops_without_more_targets(fault):
    c = FakeController()
    count = []
    def tick():
        if len(c.calls) >= 3:
            count.append(len(c.calls))
            if fault == 'stop': raise KeyboardInterrupt()
            if fault == 'mode': c.mode = 0
            if fault == 'stale': raise RuntimeError('CAN 反馈过期')
            if fault == 'scheduler': c.clock += .2
    with pytest.raises((RuntimeError, TimeoutError, KeyboardInterrupt)):
        run(c, tick)
    assert c.locked and not c.auto and count
    assert len(c.calls) == count[0]


@pytest.mark.parametrize('speed', [4., 8.])
def test_faster_placement_paths_run_without_a_false_burst_trip(speed):
    """The anti-burst bound must follow the plan's own rate, not a fixed 2 deg/s."""
    c = FakeController()
    path = np.zeros((11, 6)); path[:, 0] = np.deg2rad(np.linspace(0, 3, 11))
    times, targets = timed_path(path, speed_deg_s=speed)
    rates = np.max(np.abs(np.diff(targets, axis=0)), axis=1)/np.diff(times)
    assert rates.max() <= np.deg2rad(speed*1.001)
    events = []
    execute_path(c, times, targets, c.tick, events.append,
                 now=lambda: c.clock, wall=lambda: c.clock, sleep=c.sleep)
    assert events[-1]['event'] == 'continuous_reached' and not c.stops


def test_nonfollowing_arm_is_stopped():
    c = FakeController(); c.follow = False
    with pytest.raises(RuntimeError, match='0.8'):
        run(c, lambda: None)
    # The following-error bound now needs GUARD_PERSIST_SAMPLES consecutive
    # breaches, so a few extra targets are sent before the stop.
    assert c.locked and c.calls[-1][1][0] < np.deg2rad(1.2)


def test_single_sample_feedback_spike_does_not_stop_motion():
    """A one-sample joint-frame outlier must not abort a valid trajectory."""
    c = FakeController(); calls = {'n': 0}; real_state = c.state
    def state():
        s = real_state(); calls['n'] += 1
        if calls['n'] == 5:
            s.joints = s.joints + np.deg2rad(1.2)
        return s
    c.state = state
    times, targets, events = run(c)
    assert events[-1]['event'] == 'continuous_reached' and not c.stops


def test_motion_envelope_tolerance_is_configurable():
    """Faster paths need a wider moving band than the audited 0.3 deg."""
    c = FakeController(); seen = []
    path = np.zeros((11, 6)); path[:, 0] = np.deg2rad(np.linspace(0, 3, 11))
    times, targets = timed_path(path)
    def tick():
        seen.append(getattr(c, 'guard_tolerance_deg', None)); c.tick()
    events = []
    execute_path(c, times, targets, tick, events.append, now=lambda: c.clock,
                 wall=lambda: c.clock, sleep=c.sleep, motion_tolerance_deg=.8)
    assert .8 in seen and events[-1]['event'] == 'continuous_reached'


def test_motion_envelope_tolerance_is_validated():
    c = FakeController()
    path = np.zeros((11, 6)); path[:, 0] = np.deg2rad(np.linspace(0, 3, 11))
    times, targets = timed_path(path)
    with pytest.raises(ValueError, match='运动包络'):
        execute_path(c, times, targets, c.tick, [].append, now=lambda: c.clock,
                     wall=lambda: c.clock, sleep=c.sleep, motion_tolerance_deg=3.)


@pytest.mark.parametrize('bound,trips', [(.8, True), (1.2, False)])
def test_follow_error_bound_is_configurable(bound, trips):
    """A 0.9 deg lag trips the audited 0.8 bound but passes when widened."""
    c = FakeController(); calls = {'n': 0}; real_state = c.state
    def state():
        s = real_state(); calls['n'] += 1
        if 5 <= calls['n'] <= 12:
            s.joints = s.joints + np.deg2rad(.9)
        return s
    c.state = state
    path = np.zeros((11, 6)); path[:, 0] = np.deg2rad(np.linspace(0, 3, 11))
    times, targets = timed_path(path)
    events = []
    if trips:
        with pytest.raises(RuntimeError, match=f'{bound:g}'):
            execute_path(c, times, targets, c.tick, events.append, now=lambda: c.clock,
                         wall=lambda: c.clock, sleep=c.sleep, follow_error_deg=bound)
        assert c.locked
    else:
        execute_path(c, times, targets, c.tick, events.append, now=lambda: c.clock,
                     wall=lambda: c.clock, sleep=c.sleep, follow_error_deg=bound)
        assert events[-1]['event'] == 'continuous_reached' and not c.stops


def test_persistent_following_error_still_stops_motion():
    c = FakeController(); calls = {'n': 0}; real_state = c.state
    def state():
        s = real_state(); calls['n'] += 1
        if calls['n'] >= 5:
            s.joints = s.joints + np.deg2rad(1.2)
        return s
    c.state = state
    with pytest.raises(RuntimeError, match='0.8'):
        run(c)
    assert c.locked and not c.auto


def test_far_out_of_band_sample_stops_immediately():
    c = FakeController(); calls = {'n': 0}; real_state = c.state
    def state():
        s = real_state(); calls['n'] += 1
        if calls['n'] == 5:
            s.joints = s.joints + np.deg2rad(3.0)
        return s
    c.state = state
    with pytest.raises(RuntimeError, match='硬界限'):
        run(c)
    assert c.locked


def test_old_feedback_cannot_confirm_arrival():
    c = FakeController(); c.fresh = False
    with pytest.raises(TimeoutError, match='终点'):
        run(c)
    assert c.locked


def test_small_scheduler_delay_does_not_burst_targets():
    c = FakeController()
    def sleep(seconds): c.clock += seconds + .001
    c.sleep = sleep
    times, _, _ = run(c)
    intervals = np.diff([t for t, _ in c.calls])
    assert np.all(intervals >= np.diff(times)-1e-10)
    assert c.calls[-1][0]-c.calls[0][0] >= times[-1]


@pytest.mark.parametrize('path', [np.zeros((1, 6)), np.full((2, 6), np.nan),
                                  np.array([[0]*6, [1]*6])])
def test_invalid_plan_rejected(path):
    with pytest.raises(ValueError): timed_path(path)
