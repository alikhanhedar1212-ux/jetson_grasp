from types import SimpleNamespace as NS
import numpy as np
import pytest
from grasp import hover_observation as m


def box(xyxy, confidence=1.):
    return NS(cls=NS(item=lambda: 0), conf=NS(item=lambda: confidence),
              xyxy=[NS(cpu=lambda: NS(numpy=lambda: np.array(xyxy)))])


def sample(depth=0., count=1, boxes=None):
    boxes = [box([0, 0, 2, 2])]*count if boxes is None else boxes
    return (10., NS(depth_m=np.full((100, 100), depth)), NS(boxes=boxes), None)


@pytest.mark.parametrize('raw,category', [(0., 'zero_or_negative'), (float('nan'), 'nonfinite'),
                                         (.05, 'below_70mm'), (.7, 'above_600mm')])
def test_rejection_records_raw_depth(raw, category):
    point = m.target_point(sample(raw), {0: 'red_block'},
        lambda *a, **kw: dict(valid=False, reason='invalid_center_depth', pixel_uv=[1, 1]))
    assert point['center_depth_category'] == category
    assert point['center_raw_m'] == (raw if np.isfinite(raw) else None)


def cloud_detection(xyz, samples):
    return dict(valid=True, reason=None, xyz_camera_m=list(xyz), depth_m=xyz[2],
                pixel_uv=[10, 10], surface_samples_xyz_m=[list(s) for s in samples],
                surface_sample_count=len(samples))


def test_top_face_reference_ignores_the_camera_facing_side_wall():
    """A slanted box seen from below must still be measured on its top face."""
    side_wall = [[0., 0., z] for z in np.linspace(.170, .189, 60)]
    top_face = [[x, y, .190] for x in np.linspace(-.01, .01, 30) for y in (-.005, .005)]
    point = cloud_detection([0., 0., .180], side_wall+top_face)
    picked = m.top_face_camera_point(point, np.eye(4))
    assert picked['height_reference'] == 'top_face'
    assert picked['xyz_camera_m'][2] == pytest.approx(.190, abs=.001)
    assert picked['reference_base_z_m'] == pytest.approx(.190, abs=.001)
    assert picked['centre_reference_base_z_m'] == pytest.approx(.180)
    assert picked['reference_point_count'] >= 12


def test_top_face_reference_follows_the_camera_pose():
    """A flat top face is reported in base coordinates, not camera ones."""
    flat = [[x, y, .2] for x in np.linspace(-.02, .02, 20) for y in (-.01, .01)]
    point = cloud_detection([0., 0., .2], flat)
    T = np.eye(4); T[2, 3] = .01
    picked = m.top_face_camera_point(point, T)
    base = (T @ np.r_[np.array(picked['xyz_camera_m']), 1.])[:3]
    assert base[2] == pytest.approx(.21, abs=.001)
    # The centre is already the top face here, so the pick stays a no-op.
    assert 'height_reference' not in picked and picked['xyz_camera_m'][2] == pytest.approx(.2)


def test_top_face_reference_falls_back_without_a_sample():
    point = dict(valid=True, xyz_camera_m=[0., 0., .2], depth_m=.2, pixel_uv=[10, 10])
    assert m.top_face_camera_point(point, np.eye(4)) == point


def test_top_face_reference_needs_enough_valid_samples():
    sparse = cloud_detection([0., 0., .2], [[0., 0., .2]]*4)
    assert 'height_reference' not in m.top_face_camera_point(sparse, np.eye(4))
    broken = cloud_detection([0., 0., .2], [[np.nan, 0., .2]]*40)
    assert 'height_reference' not in m.top_face_camera_point(broken, np.eye(4))


def test_top_face_reference_rejects_an_implausible_jump():
    """A second object above the block must not become the height reference."""
    block = [[x, y, .180] for x in np.linspace(-.01, .01, 20) for y in (-.005, .005)]
    above = [[0., 0., .300]]*20
    point = cloud_detection([0., 0., .180], block+above)
    picked = m.top_face_camera_point(point, np.eye(4))
    assert 'height_reference' not in picked and picked['xyz_camera_m'] == [0., 0., .180]


def test_top_face_reference_never_rewrites_an_invalid_detection():
    point = dict(valid=False, reason='invalid_center_depth', xyz_camera_m=None,
                 surface_samples_xyz_m=[[0., 0., .3]]*40)
    assert m.top_face_camera_point(point, np.eye(4))['valid'] is False


class Preview:
    def __init__(self): self.resumes = 0; self.paused = False
    def pump(self): pass
    def sample(self): return sample(.2 if self.resumes else 0.)
    def pause_inference(self, value): self.paused = True
    def wait_until_paused(self, poll): poll()
    def resume_inference(self): self.resumes += 1; self.paused = False


class Terminal:
    """fd stand-in for the shared word reader: ready only while bytes remain."""
    def __init__(self, typed):
        self.buffer = bytearray(typed)

    def select(self, rlist, wlist, xlist, timeout=0):
        return ([rlist[0]], [], []) if self.buffer else ([], [], [])

    def read(self, fd, count):
        chunk = self.buffer[:count]
        del self.buffer[:count]
        return bytes(chunk)


def fake_terminal(monkeypatch, typed):
    """Feed the shared reader (grasp.terminal_input) from an in-memory buffer."""
    from grasp import terminal_input
    terminal = Terminal(typed)
    monkeypatch.setattr(terminal_input.select, 'select', terminal.select)
    monkeypatch.setattr(terminal_input.os, 'read', terminal.read)
    return terminal


def setup_observation(monkeypatch, keys):
    fake_terminal(monkeypatch, keys)
    monkeypatch.setattr(m, 'poll_key', lambda fd: None)
    monkeypatch.setattr(m.time, 'monotonic', lambda: 10.)
    controller = NS(locked=False, tick=lambda: None)
    preview = Preview(); events = []
    def locate(frame, *a, **kw):
        if frame.depth_m[1, 1] == .2:
            return dict(valid=True, xyz_camera_m=[0, 0, .2])
        return dict(valid=False, reason='invalid_center_depth', pixel_uv=[1, 1])
    def read(): return NS(joints=np.zeros(6)), np.eye(4), None
    return controller, preview, events, locate, read


def test_retry_prompt_can_be_edited_before_enter(monkeypatch, capsys):
    """Regression: the grasp leg's retry prompt had no visible Backspace/Delete.

    Typing "retrry", fixing it with Backspace and the Delete key (`ESC [ 3 ~`) and
    then retyping the missing letters must be accepted -- and the Delete key must
    not stop the session through its leading Escape.
    """
    # "retrry" -> Backspace -> Delete -> "y" = "retry"
    c, p, events, locate, read = setup_observation(monkeypatch, b'retrry\x7f\x1b[3~y\n')
    result = m.observe_at_fixed(c, 1, p, read, {0: 'red_block'}, locate, events.append)
    assert result is not None and p.resumes == 1
    assert events[-1]['event'] == 'observation_retry'
    out = capsys.readouterr().out
    assert out.count('\b \b') == 2


def test_invalid_depth_then_retry_uses_new_valid_observation(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'go\nretry\n')
    result = m.observe_at_fixed(c, 1, p, read, {0: 'red_block'}, locate, events.append)
    assert result[3]['valid'] and result[0][1].depth_m[1, 1] == .2
    assert p.resumes == 1
    assert [e['event'] for e in events] == ['observation_rejected', 'observation_retry']


def test_invalid_depth_quit_does_not_stop_or_move(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'quit\n')
    assert m.observe_at_fixed(c, 1, p, read, {0: 'red_block'}, locate, events.append) is None
    assert p.paused and not c.locked and p.resumes == 0
    assert events[-1]['event'] == 'observation_exit'


def test_invalid_detection_accepts_home_without_retrying(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'home\n')
    calls = []
    assert m.observe_at_fixed(c, 1, p, read, {0: 'red_block'}, locate, events.append,
                              home=lambda: calls.append('home')) is None
    assert calls == ['home']
    assert p.resumes == 0
    assert [row['event'] for row in events][-2:] == [
        'observation_home_requested', 'observation_home_reached']


def test_tracking_retry_keeps_detector_running(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'retry\n')
    frames = iter([sample(0.), sample(.2)])
    p.sample = lambda: next(frames)
    result = m.observe_at_fixed(c, 1, p, read, {0: 'red_block'}, locate, events.append,
                                continuous_inference=True)
    assert result[3]['valid'] and not p.paused and p.resumes == 0


def test_tracking_plan_retry_does_not_resume_an_already_running_detector(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'retry\n')
    assert m.wait_for_plan_retry(c, 1, p, read, read()[0], events.append,
                                 continuous_inference=True)
    assert not p.paused and p.resumes == 0


def test_planning_rejection_accepts_home_without_retrying_detector(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'home\n')
    calls = []
    assert not m.wait_for_plan_retry(c, 1, p, read, read()[0], events.append,
                                     continuous_inference=True,
                                     home=lambda: calls.append('home'))
    assert calls == ['home']
    assert p.resumes == 0
    assert [row['event'] for row in events] == [
        'planning_home_requested', 'planning_home_reached']


@pytest.mark.parametrize('key', [b' ', b'\x1b', b'\x03'])
def test_retry_wait_preserves_manual_stop(monkeypatch, key):
    c, p, events, locate, read = setup_observation(monkeypatch, key)
    with pytest.raises(KeyboardInterrupt):
        m.observe_at_fixed(c, 1, p, read, {0: 'red_block'}, locate, events.append)


def test_retry_wait_preserves_can_failure(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'retry\n')
    def tick():
        if p.paused: raise RuntimeError('CAN 反馈过期')
    c.tick = tick
    with pytest.raises(RuntimeError, match='CAN 反馈过期'):
        m.observe_at_fixed(c, 1, p, read, {0: 'red_block'}, locate, events.append)
    assert p.resumes == 0


def test_missing_block_is_recoverable():
    def locate(*a, **kw): raise AssertionError('cannot locate ambiguous box')
    point = m.target_point(sample(count=0), {0: 'red_block'}, locate)
    assert not point['valid'] and point['reason'] == 'red_block_count'


def test_overlapping_duplicate_boxes_keep_highest_confidence():
    boxes = [box([10, 10, 30, 30], .7), box([11, 11, 31, 31], .9)]
    seen = []
    point = m.target_point(sample(.2, boxes=boxes), {0: 'red_block'},
                           lambda frame, xyxy, **kw: (seen.append(xyxy.copy()) or
                                                      dict(valid=True, xyz_camera_m=[0, 0, .2])))
    assert point['valid'] and point['count'] == 2 and point['deduplicated_count'] == 1
    assert point['selection'] == 'overlap_deduplicated'
    np.testing.assert_equal(seen[0], [11, 11, 31, 31])


def test_distinct_boxes_follow_previous_target_only_when_unambiguous():
    boxes = [box([12, 10, 32, 30], .7), box([70, 70, 90, 90], .95)]
    point = m.target_point(sample(.2, boxes=boxes), {0: 'red_block'},
                           lambda frame, xyxy, **kw: dict(valid=True, xyz_camera_m=[0, 0, .2]),
                           previous_box_xyxy=[10, 10, 30, 30])
    assert point['valid'] and point['selection'] == 'previous_box_association'
    np.testing.assert_equal(point['selected_box_xyxy'], [12, 10, 32, 30])


def test_first_frame_with_distinct_boxes_remains_ambiguous():
    boxes = [box([10, 10, 30, 30], .9), box([70, 70, 90, 90], .8)]
    point = m.target_point(sample(.2, boxes=boxes), {0: 'red_block'},
                           lambda *a, **kw: pytest.fail('ambiguous target must not be located'))
    assert not point['valid'] and point['selection'] == 'ambiguous'


@pytest.mark.parametrize('keys,retry', [(b'go\nretry\n', True), (b'quit\n', False), (b'\x04', False)])
def test_plan_rejection_waits_for_explicit_retry_or_exit(monkeypatch, keys, retry):
    c, p, events, locate, read = setup_observation(monkeypatch, keys)
    p.paused = True
    anchor, _, _ = read()
    assert m.wait_for_plan_retry(c, 1, p, read, anchor, events.append) is retry
    assert p.resumes == int(retry)
    assert events[-1]['event'] == ('planning_retry' if retry else 'planning_exit')


def test_plan_retry_preserves_position_guard(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'retry\n')
    anchor, _, _ = read()
    def moved(): return NS(joints=np.full(6, np.deg2rad(.9))), np.eye(4), None
    with pytest.raises(RuntimeError, match='偏离固定点'):
        m.wait_for_plan_retry(c, 1, p, moved, anchor, events.append)
    assert p.resumes == 0


def test_hold_snap_does_not_stop_the_wait(monkeypatch):
    """The documented 0.3-0.4 deg snap after ~12-16 s must stay recoverable."""
    c, p, events, locate, read = setup_observation(monkeypatch, b'retry\n')
    anchor, _, _ = read()
    def snapped(): return NS(joints=np.full(6, np.deg2rad(.4))), np.eye(4), None
    assert m.wait_for_plan_retry(c, 1, p, snapped, anchor, events.append) is True
    assert events[-1]['event'] == 'planning_retry'


def test_observation_tolerates_the_same_snap(monkeypatch):
    c, p, events, locate, read = setup_observation(monkeypatch, b'go\nretry\n')
    def snapped(): return NS(joints=np.full(6, np.deg2rad(.4))), np.eye(4), None
    result = m.observe_at_fixed(c, 1, p, snapped, {0: 'red_block'}, locate, events.append)
    assert result[3]['valid'] and not c.locked


@pytest.mark.parametrize('fault', ['can', 'stop'])
def test_plan_retry_preserves_fault_and_manual_stop(monkeypatch, fault):
    c, p, events, locate, read = setup_observation(monkeypatch, b' ')
    anchor, _, _ = read()
    if fault == 'can':
        def tick(): raise RuntimeError('CAN 反馈过期')
        c.tick = tick
    with pytest.raises(RuntimeError if fault == 'can' else KeyboardInterrupt):
        m.wait_for_plan_retry(c, 1, p, read, anchor, events.append)
    assert p.resumes == 0
