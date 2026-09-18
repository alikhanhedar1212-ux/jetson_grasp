import struct
import numpy as np
import pytest
from grasp import j6_experiment as j6
from test_arm_console import Arm, Receiver, frames


def test_plan_preserves_first_five_and_checks_all_targets():
    limits = np.deg2rad([[-180, 180]] * 6)
    targets = j6.make_targets(j6.TARGET_DEG, [0, 10, 0, -10, 0], limits)
    np.testing.assert_allclose(targets[:, :5], np.tile(targets[0, :5], (5, 1)))
    np.testing.assert_allclose(np.rad2deg(targets[:, 5]), np.array([0, 10, 0, -10, 0]) - .902)
    for offsets in ([0, 30, 0], [1, 0], [0, float('nan'), 0], [0, 10]):
        with pytest.raises(ValueError):
            j6.make_targets(j6.TARGET_DEG, offsets, limits)
    with pytest.raises(ValueError):
        j6.make_targets([0, 0, 0, 0, 0, 179], [0, 5, 0], limits)


@pytest.mark.parametrize('failure', [None, 'drift', 'stale', 'interrupt', 'mode', 'no_arrival'])
def test_sequence_and_failure_stop(tmp_path, monkeypatch, failure):
    clock = [100.]
    monkeypatch.setattr(j6.time, 'time', lambda: clock[0])
    monkeypatch.setattr(j6.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(j6.time, 'sleep', lambda t: clock.__setitem__(0, clock[0] + t))
    arm, receiver = Arm(), Receiver()
    c = j6.Controller(arm, receiver, tmp_path / 'unused.json', {},
                      np.deg2rad([[-180, 180]] * 6), speed=3, timeout=5, emit=lambda _: None)
    home = np.deg2rad([0, 10, -10, 0, 20, 0])
    offsets = [0, 5, 0, -5, 0]
    targets = j6.make_targets(j6.TARGET_DEG, offsets, c.limits)
    samples = []
    def poll():
        moves = [item[1] for item in arm.calls if isinstance(item, tuple) and item[0] == 'move']
        q = np.array(moves[-1]) if moves else np.deg2rad([0, 10, -10, 0, 0, 0])
        if samples:
            if failure == 'drift':
                q[0] += np.deg2rad(.4)
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            if failure == 'no_arrival':
                q[5] += np.deg2rad(2)
        receiver.frames = frames(clock[0], enabled='enable' in arm.calls,
                                 joints=tuple(np.rint(np.rad2deg(q) * 1000).astype(int)))
        if samples and failure == 'mode':
            receiver.frames[0x2A1] = (bytes([1, 0, 0, 0, 0, 0, 0, 0]), clock[0])
        for ident in (0x2A2, 0x2A3, 0x2A4):
            stamp = clock[0] - 1 if samples and failure == 'stale' else clock[0]
            receiver.frames[ident] = (struct.pack('>ii', 0, 0), stamp)
    poll()
    if failure:
        with pytest.raises((RuntimeError, KeyboardInterrupt, TimeoutError)):
            j6.run_sequence(c, targets, offsets, .5, poll, samples.append, lambda *args: [0.] * 6, home_target=home)
        assert c.locked and arm.calls[-1] == 'stop'
        assert len(samples) == 1
    else:
        j6.run_sequence(c, targets, offsets, .5, poll, samples.append, lambda *args: [0.] * 6, home_target=home)
        assert len(samples) == 5 and not c.locked
        moves = [item[1] for item in arm.calls if isinstance(item, tuple) and item[0] == 'move']
        # First target is a holding seed before enabling, then fixed point, then sweep.
        assert len(moves) == 7
        for q in moves[2:-1]:
            np.testing.assert_allclose(q[:5], targets[0, :5])
        np.testing.assert_allclose(moves[-2], targets[0])
        np.testing.assert_allclose(moves[-1], home)
        assert c.fixed_anchor is None
        assert all('piper_x' in s and 'piper' in s for s in samples)
    assert not c.home_file.exists()


def test_camera_hook_runs_before_home_and_invalid_board_still_returns(tmp_path, monkeypatch):
    # Reuse the sequence test's simulated robot while injecting a camera hook.
    original = j6.run_sequence
    observed = []
    def wrapped(controller, targets, offsets, dwell, poll, save, fk, home_target=None):
        def observe(sample, target, step):
            step()
            assert controller.fixed_anchor is not None
            np.testing.assert_allclose(controller.state().joints, target, atol=np.deg2rad(.001))
            observed.append(sample['index'])
            return {'valid': False, 'error': 'board outside image'}
        return original(controller, targets, offsets, dwell, poll, save, fk,
                        home_target=home_target, observe=observe)
    monkeypatch.setattr(j6, 'run_sequence', wrapped)
    test_sequence_and_failure_stop(tmp_path, monkeypatch, None)
    assert observed == list(range(5))


def test_firmware_waits_for_receive_and_retries(monkeypatch):
    from types import SimpleNamespace
    clock = [0.]
    calls = []
    monkeypatch.setattr(j6.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(j6.time, 'sleep', lambda t: clock.__setitem__(0, clock[0]+t))
    def query(**kwargs):
        assert clock[0] >= .2
        calls.append(clock[0])
        return None if len(calls) == 1 else {'software_version':'test'}
    arm = SimpleNamespace(get_fps=lambda: 100 if clock[0] >= .2 else 0, get_firmware=query)
    assert j6.query_firmware(arm)['software_version'] == 'test'
    assert len(calls) == 2
    arm.get_fps = lambda: 0
    with pytest.raises(RuntimeError, match='接收帧率'):
        j6.query_firmware(arm)
    assert len(calls) == 2
