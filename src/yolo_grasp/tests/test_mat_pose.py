import json
import struct
from types import SimpleNamespace
import numpy as np
import pytest
from grasp import mat_pose as mat
from test_arm_console import Arm, Receiver, frames


@pytest.mark.parametrize('failure', [None, 'arm_fault', 'grip_timeout', 'grip_fault', 'escape', 'stale_grip'])
def test_move_then_open_with_live_feedback(tmp_path, monkeypatch, failure):
    clock = [100.]
    monkeypatch.setattr(mat.time, 'time', lambda: clock[0])
    monkeypatch.setattr(mat.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(mat.time, 'sleep', lambda t: clock.__setitem__(0, clock[0] + t))
    arm, receiver = Arm(), Receiver()
    c = mat.FixedController(arm, receiver, tmp_path / 'home.json', {}, [[-3.14, 3.14]] * 6, speed=3, emit=lambda _: None)
    c.home_file.write_text('preserve me')
    grip_calls = []
    def open_grip(**kwargs):
        assert c.active is None and not c.locked
        np.testing.assert_allclose(c.state().joints, np.deg2rad(mat.TARGET_DEG), atol=1e-6)
        grip_calls.append(kwargs)
    gripper = SimpleNamespace(move_gripper_m=open_grip, disable_gripper=lambda: grip_calls.append('disable'))
    def poll():
        moves = [v for v in arm.calls if isinstance(v, tuple) and v[0] == 'move']
        joints = tuple(round(x * 1000) for x in mat.TARGET_DEG) if len(moves) > 1 else (0, 10000, -10000, 0, 0, 0)
        receiver.frames = frames(clock[0], joints=joints, enabled='enable' in arm.calls)
        if failure == 'arm_fault' and len(moves) > 1:
            receiver.frames[0x2A1] = (bytes([1, 7, 1, 0, 0, 0, 0, 0]), clock[0])
        width = 50000 if grip_calls and failure != 'grip_timeout' else 20000
        flags = 0x41 if grip_calls and failure == 'grip_fault' else 0x40
        stamp = clock[0] - 1 if grip_calls and failure == 'stale_grip' else clock[0]
        receiver.frames[0x2A8] = (struct.pack('>ihBB', width, 0, flags, 0), stamp)
        if failure == 'escape' and grip_calls:
            raise KeyboardInterrupt()
    poll()
    if failure:
        with pytest.raises((RuntimeError, TimeoutError, KeyboardInterrupt)):
            mat.run_sequence(c, gripper, .05, 1, poll)
        assert c.locked and arm.calls[-1] == 'stop'
        if failure == 'arm_fault':
            assert grip_calls == []
        else:
            assert grip_calls[-1] == 'disable'
    else:
        mat.run_sequence(c, gripper, .05, 1, poll)
        assert grip_calls == [{'value': .05, 'force': 1}]
        assert not c.locked and c.monitor
    assert c.home_file.read_text() == 'preserve me'


def test_open_cannot_close_or_use_stale_feedback(monkeypatch):
    monkeypatch.setattr(mat.time, 'time', lambda: 100.)
    receiver = Receiver()
    receiver.frames[0x2A8] = (struct.pack('>ihBB', 60000, 0, 0, 0), 100.)
    with pytest.raises(ValueError, match='拒绝闭合'):
        mat.check_opening(receiver, .05)
    receiver.frames[0x2A8] = (struct.pack('>ihBB', 20000, 0, 0, 0), 99.)
    with pytest.raises(RuntimeError, match='过期'):
        mat.check_opening(receiver, .05)


@pytest.mark.parametrize('value', [.07, .1, 0, .08, float('nan'), None])
def test_configured_max_width(value):
    gripper = SimpleNamespace(get_gripper_teaching_pendant_param=lambda **kw:
        None if value is None else SimpleNamespace(msg=SimpleNamespace(max_range_config=value)))
    if value in (.07, .1):
        assert mat.configured_max_width(gripper) == value
    else:
        with pytest.raises(RuntimeError):
            mat.configured_max_width(gripper)


def test_space_without_enter_interrupts(monkeypatch):
    monkeypatch.setattr(mat.select, 'select', lambda *args: ([123], [], []))
    monkeypatch.setattr(mat.os, 'read', lambda *args: b' ')
    with pytest.raises(mat.UserStop):
        mat.poll_key(123)


def test_home_is_saved_pose_not_mat(tmp_path):
    c = mat.FixedController(Arm(), Receiver(), tmp_path / 'home.json', {}, [[-3.14, 3.14]] * 6)
    target = np.deg2rad([0, 10, -10, 0, 0, 0])
    c.save_home(target)
    np.testing.assert_allclose(c.load_home(), target)
    c.mat_target = True
    np.testing.assert_allclose(c.load_home(), np.deg2rad(mat.TARGET_DEG))
    c.mat_target = False
    c.locked = True
    with pytest.raises(RuntimeError, match='锁定'):
        c.command('home')
    assert c.arm.calls == []


def test_space_cancels_sequence_without_opening(tmp_path, monkeypatch):
    monkeypatch.setattr(mat.time, 'time', lambda: 100.)
    r = Receiver()
    r.frames[0x2A8] = (struct.pack('>ihBB', 20000, 0, 0, 0), 100.)
    a = Arm()
    c = mat.FixedController(a, r, tmp_path / 'home.json', {}, [[-3.14, 3.14]] * 6, emit=lambda _: None)
    def interrupt():
        raise mat.UserStop('space')
    # Gripper has no motion method: cancellation must never invoke one.
    with pytest.raises(mat.UserStop):
        mat.run_sequence(c, SimpleNamespace(), .05, 1, interrupt)
    assert c.locked and a.calls == ['stop']


@pytest.mark.parametrize('failure', [None, 'enabled_early', 'drift', 'mode_timeout', 'seed_write'])
def test_disabled_preparation_order_and_failures(tmp_path, monkeypatch, failure):
    clock = [100.]
    monkeypatch.setattr(mat.time, 'time', lambda: clock[0])
    monkeypatch.setattr(mat.time, 'monotonic', lambda: clock[0])
    a, r = Arm(), Receiver()
    r.frames = frames(clock[0], enabled=False)
    c = mat.FixedController(a, r, tmp_path / 'home.json', {}, [[-3.14, 3.14]] * 6, speed=3, emit=lambda _: None)
    if failure == 'seed_write':
        def fail(target):
            raise RuntimeError('seed write failed')
        a.move_j = fail
    c.command('enable')
    assert a.calls == []
    for i in range(1, 55):
        clock[0] = 100 + i * .1
        prepared = c.phase == '失能目标准备'
        enabled = 'enable' in a.calls or (prepared and failure == 'enabled_early')
        joints = (0, 10000, -10000, 0, 0, 500 if prepared and failure == 'drift' else 0)
        r.frames = frames(clock[0], enabled=enabled, joints=joints)
        if prepared and failure == 'mode_timeout':
            r.frames[0x2A1] = (bytes(8), clock[0])
        try:
            c.tick()
        except RuntimeError:
            break
        except TimeoutError:
            break
        if c.active is None:
            break
    if failure:
        assert c.locked and 'enable' not in a.calls
        assert a.calls[-1] == 'stop'
        assert a.auto_mode
    else:
        assert c.active is None and c.prepared and not c.locked
        assert [v[0] if isinstance(v, tuple) else v for v in a.calls] == ['move', 'speed', 'mode', 'enable']
        count = len(a.calls)
        c.move_to_mat()
        assert len(a.calls) == count  # No speed/mode/target frames until fresh settled feedback.
        for _ in range(8):
            clock[0] += .1
            r.frames = frames(clock[0])
            c.tick()
        assert [v[0] if isinstance(v, tuple) else v for v in a.calls[count:]] == ['move']


def test_prepare_rejects_already_enabled_arm(tmp_path, monkeypatch):
    monkeypatch.setattr(mat.time, 'time', lambda: 100.)
    a, r = Arm(), Receiver()
    c = mat.FixedController(a, r, tmp_path / 'home.json', {}, [[-3.14, 3.14]] * 6)
    with pytest.raises(RuntimeError, match='六轴失能'):
        c.command('enable')
    assert a.calls == []


def fixed_controller(tmp_path, arm, receiver, speed=3):
    controller = mat.FixedController(arm, receiver, tmp_path / 'home.json', {},
                                     [[-3.14, 3.14]] * 6, speed=speed, emit=lambda _: None)
    controller.home_file.write_text(json.dumps(
        {'schema_version': 1, 'joint_radians': [.1, 1., -.9, -.02, .6, -.01]}))
    return controller


def test_adopt_enabled_session_makes_home_reachable_without_disable(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(mat.time, 'time', lambda: clock[0])
    monkeypatch.setattr(mat.time, 'monotonic', lambda: clock[0])
    arm, receiver = Arm(), Receiver()
    arm._msg_mode = SimpleNamespace(move_spd_rate_ctrl=50)
    controller = fixed_controller(tmp_path, arm, receiver)
    assert controller.enable_needed(), 'a fresh session still needs the enable prep'
    receiver.frames = frames(clock[0], joints=tuple(round(v*1000) for v in mat.TARGET_DEG), enabled=True)
    controller.adopt_enabled_session()
    assert controller.prepared and not controller.enable_needed()
    assert arm._msg_mode.move_spd_rate_ctrl == 3, 'speed must ride the target frame'
    assert arm.calls == [], 'adopting a session must not send anything'


@pytest.mark.parametrize('enabled,mode,ctrl,message', [
    (False, 1, 1, '未全部使能'),
    (True, 2, 1, 'CAN/J'),
    (True, 1, 0, 'CAN/J'),
])
def test_adopt_rejects_sessions_it_cannot_trust(tmp_path, monkeypatch, enabled, mode, ctrl, message):
    clock = [100.]
    monkeypatch.setattr(mat.time, 'time', lambda: clock[0])
    monkeypatch.setattr(mat.time, 'monotonic', lambda: clock[0])
    arm, receiver = Arm(), Receiver()
    arm._msg_mode = SimpleNamespace(move_spd_rate_ctrl=50)
    controller = fixed_controller(tmp_path, arm, receiver)
    receiver.frames = frames(clock[0], joints=tuple(round(v*1000) for v in mat.TARGET_DEG),
                             enabled=enabled)
    status = bytes([ctrl, 0, mode, 0, 0, 0, 0, 0])
    receiver.frames[0x2A1] = (status, clock[0])
    with pytest.raises(RuntimeError, match=message):
        controller.adopt_enabled_session()
    assert not controller.prepared


def test_adopted_session_reaches_home_without_disable(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(mat.time, 'time', lambda: clock[0])
    monkeypatch.setattr(mat.time, 'monotonic', lambda: clock[0])
    arm, receiver = Arm(), Receiver()
    arm._msg_mode = SimpleNamespace(move_spd_rate_ctrl=50)
    controller = fixed_controller(tmp_path, arm, receiver)
    receiver.frames = frames(clock[0], joints=tuple(round(v*1000) for v in mat.TARGET_DEG),
                             enabled=True)
    controller.adopt_enabled_session()
    home_deg = np.rad2deg(controller.load_home())
    controller.command('home')
    sent = False
    for _ in range(200):
        clock[0] += .1
        joints = home_deg if sent else mat.TARGET_DEG
        receiver.frames = frames(clock[0], joints=tuple(round(v*1000) for v in joints),
                                 enabled=True)
        controller.tick()
        sent = sent or any(isinstance(call, tuple) and call[0] == 'move' for call in arm.calls)
        if controller.active is None:
            break
    assert sent, 'an adopted session must be able to send the home target'
    assert controller.active is None and not controller.locked
    assert 'enable' not in arm.calls
