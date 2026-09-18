import copy
import json
import os
import signal
import sys
from types import SimpleNamespace
import struct
import termios
from contextlib import contextmanager

import numpy as np
import pytest

from grasp import arm_console as ac


def frames(stamp=100., enabled=True, joints=(0, 10000, -10000, 0, 0, 0)):
    result = {0x2A1: (bytes([1, 0, 1, 0, 0, 0, 0, 0]), stamp)}
    for index, can_id in enumerate(ac.JOINT_IDS):
        result[can_id] = (struct.pack(">ii", *joints[index * 2:index * 2 + 2]), stamp)
    for can_id in ac.DRIVER_IDS:
        result[can_id] = (bytes([0, 0, 0, 0, 0, 0x40 if enabled else 0, 0, 0]), stamp)
    return result


class Arm:
    def __init__(self):
        self.calls = []
        self.auto_mode = True

    def enable(self):
        self.calls.append("enable")
        return True  # Cached ACK must not complete an operation.

    def disable(self):
        self.calls.append("disable")
        return True

    def electronic_emergency_stop(self):
        self.calls.append("stop")

    def set_speed_percent(self, speed):
        self.calls.append(("speed", speed))

    def set_motion_mode(self, mode):
        self.calls.append(("mode", mode))

    def get_auto_set_motion_mode_enabled(self):
        return self.auto_mode

    def set_auto_set_motion_mode_enabled(self, enabled):
        self.auto_mode = enabled

    def move_j(self, target):
        self.calls.append(("move", target))


class Receiver:
    def __init__(self):
        self.frames = frames()

    def snapshot(self):
        return copy.deepcopy(self.frames)


@pytest.fixture
def rig(tmp_path, monkeypatch):
    clock = [100.]
    monkeypatch.setattr(ac.time, "time", lambda: clock[0])
    monkeypatch.setattr(ac.time, "monotonic", lambda: clock[0])
    arm, receiver = Arm(), Receiver()
    controller = ac.Controller(arm, receiver, tmp_path / "home.json",
                               {"model": "piper_x", "channel": "can0", "firmware": "S-V1.8-9"},
                               [[-3.14, 3.14]] * 6, emit=lambda s: None)

    def advance(delta=.1, **kwargs):
        clock[0] += delta
        receiver.frames = frames(clock[0], **kwargs)
        controller.tick()

    return controller, arm, receiver, clock, advance


def test_decode_joint_units_and_driver_faults():
    f = frames()
    f[0x263] = (bytes([0, 0, 0, 0, 0, 0xC0, 0, 0]), 100.)
    s = ac.decode_state(f, 100.1)
    assert s.joints[1] == pytest.approx(np.deg2rad(10))
    assert s.joints[2] == pytest.approx(np.deg2rad(-10))
    assert all(s.enabled)
    with pytest.raises(RuntimeError, match="故障"):
        s.healthy()


@pytest.mark.parametrize("kind", ["missing", "stale", "future", "skew", "short"])
def test_bad_feedback_rejected(kind):
    f = frames()
    if kind == "missing":
        del f[0x262]
    elif kind == "short":
        f[0x2A5] = (b"short", 100.)
    else:
        f[0x2A5] = (f[0x2A5][0], {"stale": 99., "future": 101., "skew": 99.9}[kind])
    with pytest.raises(RuntimeError):
        ac.decode_state(f, 100.1)


def test_save_home_requires_stable_interval_and_sends_nothing(rig):
    c, arm, _, _, advance = rig
    c.command("set-home")
    advance()
    assert not c.home_file.exists()
    for _ in range(7):
        advance()
    assert c.active is None
    assert c.load_home()[1] == pytest.approx(np.deg2rad(10))
    assert arm.calls == []


@pytest.mark.parametrize("enabled", [False, True])
def test_rejected_home_save_preserves_file_and_session(rig, enabled):
    c, arm, receiver, clock, advance = rig
    c.save_home(np.zeros(6))
    original = c.home_file.read_bytes()
    c.limits[1] = [0, np.pi]
    c.limits[2] = [-np.deg2rad(170), 0]
    messages = []
    c.emit = messages.append
    invalid = (0, -850, 1490, 0, 0, 0)
    receiver.frames = frames(clock[0], enabled=enabled, joints=invalid)
    c.command("set-home")
    for _ in range(8):
        advance(enabled=enabled, joints=invalid)
    assert c.active is None and not c.locked
    assert arm.calls == []
    assert c.home_file.read_bytes() == original
    assert "第 2 轴 -0.850°" in messages[-1]
    assert "第 3 轴 1.490°" in messages[-1]
    assert "允许范围" in messages[-1]
    # A subsequent valid save works without restarting the controller.
    valid = (0, 10000, -10000, 0, 0, 0)
    receiver.frames = frames(clock[0], enabled=enabled, joints=valid)
    c.command("set-home")
    for _ in range(8):
        advance(enabled=enabled, joints=valid)
    assert c.active is None and not c.locked and arm.calls == []
    assert c.load_home()[1] == pytest.approx(np.deg2rad(10))


def test_zero_home_only_saves_and_cannot_overwrite_during_motion(rig):
    c, arm, _, _, advance = rig
    c.command("zero-home")
    for _ in range(7):
        advance()
    np.testing.assert_array_equal(c.load_home(), np.zeros(6))
    assert arm.calls == []
    c.command("home")
    with pytest.raises(RuntimeError, match="等待"):
        c.command("set-home")


def test_enabled_idle_fault_also_requests_stop(rig):
    c, arm, receiver, *_ = rig
    c.tick()
    receiver.frames[0x2A1] = (bytes([0, 7, 0, 0, 0, 0, 0, 0]), 100.)
    c.tick()
    assert c.locked and arm.calls == ["stop"]


def test_arm_cli_bypasses_camera_configuration(monkeypatch):
    from grasp.__main__ import main
    calls = []
    monkeypatch.setattr(ac, "main", lambda argv: calls.append(argv) or 0)
    assert main(["--config", "/does/not/exist.json", "arm", "--channel", "can1"]) == 0
    assert calls[0][:2] == ["--channel", "can1"]


def test_home_refuses_wrong_identity_and_nonfinite_targets(rig):
    c, arm, *_ = rig
    c.save_home([0] * 6)
    data = json.loads(c.home_file.read_text())
    data["channel"] = "can1"
    c.home_file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="不匹配"):
        c.command("home")
    assert not arm.calls
    with pytest.raises(ValueError):
        c.save_home([float("nan")] * 6)
    with pytest.raises(ValueError):
        c.save_home([10.] * 6)


def test_enable_disable_require_fresh_six_driver_feedback(rig):
    c, arm, receiver, _, advance = rig
    receiver.frames = frames(enabled=False)
    c.command("enable")
    c.tick()
    assert c.active == "enable"
    advance(enabled=True)
    assert c.active == "enable"
    for _ in range(7):
        advance(enabled=True)
    assert c.active is None
    c.command("disable")
    c.tick()
    assert c.active == "disable"
    advance(enabled=False)
    assert c.active is None
    assert arm.calls == ["enable", "disable"]


def test_home_needs_enabled_arm_and_post_command_arrival(rig):
    c, arm, receiver, _, advance = rig
    c.save_home([0] * 6)
    receiver.frames = frames(enabled=False)
    with pytest.raises(RuntimeError, match="enable"):
        c.command("home")
    receiver.frames = frames()
    c.command("home")
    c.command("home")  # A repeated command must not restart the timeout.
    for _ in range(8):
        advance()
    assert len([v for v in arm.calls if isinstance(v, tuple) and v[0] == "move"]) == 1
    for _ in range(7):
        advance()  # Reports arrived, but still at the wrong angles.
    assert c.active == "home"
    for _ in range(7):
        advance(joints=(0,) * 6)
    assert c.active is None
    assert "stop" not in arm.calls


@pytest.mark.parametrize("command", ["home", "jog 6 2"])
@pytest.mark.parametrize("failure", ["fault", "stale", "lost_enable", "timeout"])
def test_motion_failures_stop_and_latch_without_auto_return(rig, failure, command):
    c, arm, receiver, clock, advance = rig
    c.save_home([0] * 6)
    c.command(command)
    if failure == "fault":
        receiver.frames[0x2A1] = (bytes([0, 7, 0, 0, 0, 0, 0, 0]), clock[0])
        c.tick()
    elif failure == "stale":
        clock[0] += 1
        c.tick()
    elif failure == "lost_enable":
        advance(enabled=False)
    else:
        advance(31)
    assert c.locked and c.active is None
    assert arm.calls[-1] == "stop"
    with pytest.raises(RuntimeError, match="锁定"):
        c.command("home")
    with pytest.raises(RuntimeError, match="锁定"):
        c.command("enable")
    c.command("disable")
    advance(enabled=False)
    assert c.active is None


def test_partial_send_failure_stops_and_preserves_exception(rig):
    c, arm, _, _, advance = rig
    c.save_home([0] * 6)

    def fail(target):
        raise OSError("partial CAN write")

    arm.move_j = fail
    c.command("home")
    for _ in range(8):
        advance()
    assert arm.auto_mode
    assert c.locked and arm.calls[-1] == "stop"


def test_stop_send_failure_still_latches(rig):
    c, arm, *_ = rig

    def fail():
        raise OSError("CAN disconnected")

    arm.electronic_emergency_stop = fail
    c.stop("test")
    assert c.locked and c.active is None


def test_space_and_escape_handled_without_newline(monkeypatch):
    keys = iter([b" ", b"\x1b", b"\x04"])
    calls = []

    @contextmanager
    def keyboard():
        yield 123

    class ConsoleController:
        def tick(self):
            calls.append("tick")

        def command(self, cmd):
            calls.append(cmd)

        def stop(self, reason):
            calls.append("stop")

    monkeypatch.setattr(ac, "keyboard", keyboard)
    monkeypatch.setattr(ac.select, "select", lambda *a: ([123], [], []))
    monkeypatch.setattr(ac.os, "read", lambda *a: next(keys))
    ac.console(ConsoleController())
    assert calls == ["tick", "tick", "stop", "tick"]


@pytest.mark.parametrize("delta", [2, 20, -20])
def test_jog_target_and_arrival_preserve_home(rig, delta):
    c, arm, _, _, advance = rig
    c.save_home(np.zeros(6))
    saved = c.home_file.read_bytes()
    c.command(f"jog 6 {delta}")
    assert arm.calls == [("speed", 3), ("mode", "j")]
    for _ in range(8):
        advance()
    np.testing.assert_allclose(arm.calls[2][1], np.deg2rad([0, 10, -10, 0, 0, delta]))
    for _ in range(8):
        advance()
    assert c.active == "jog"
    for _ in range(8):
        advance(joints=(0, 10000, -10000, 0, 0, delta * 1000))
    assert c.active is None and not c.locked
    assert c.home_file.read_bytes() == saved


@pytest.mark.parametrize("command", ["jog", "jog 0 1", "jog 7 1", "jog 6 0", "jog 6 20.1", "jog 5 2.1", "jog 6 nan", "jog 6 inf"])
def test_jog_bad_input_never_sends(rig, command):
    c, arm, *_ = rig
    with pytest.raises(ValueError):
        c.command(command)
    assert arm.calls == []


@pytest.mark.parametrize("reason", ["current_limit", "target_limit", "disabled"])
def test_jog_preconditions_never_send(rig, reason):
    c, arm, receiver, clock, _ = rig
    if reason == "current_limit":
        c.limits[1] = [0, .1]
    elif reason == "target_limit":
        c.limits[5] = [-.01, .01]
    elif reason == "disabled":
        receiver.frames = frames(clock[0], enabled=False)
    else:
        receiver.frames[0x2A1] = (bytes([0, 0, 0, 0, 1, 0, 0, 0]), clock[0])
    with pytest.raises((ValueError, RuntimeError)):
        c.command("jog 6 2")
    assert arm.calls == []


def test_jog_spaces_do_not_trigger_home(monkeypatch):
    keys = iter(bytes([c]) for c in b" jog 6 2\n\x04")
    calls = []
    @contextmanager
    def keyboard():
        yield 123
    monkeypatch.setattr(ac, "keyboard", keyboard)
    monkeypatch.setattr(ac.select, "select", lambda *a: ([123], [], []))
    monkeypatch.setattr(ac.os, "read", lambda *a: next(keys))
    ac.console(SimpleNamespace(tick=lambda: None, command=calls.append, stop=calls.append))
    assert calls == ["jog 6 2"]


def test_terminal_restored_after_exception(monkeypatch):
    master, slave = os.openpty()
    try:
        with os.fdopen(os.dup(slave)) as stream:
            monkeypatch.setattr(ac.sys, "stdin", stream)
            original = termios.tcgetattr(slave)
            with pytest.raises(RuntimeError):
                with ac.keyboard():
                    assert not termios.tcgetattr(slave)[3] & termios.ICANON
                    assert not termios.tcgetattr(slave)[0] & (termios.IXON | termios.IXOFF)
                    raise RuntimeError("exit")
            assert termios.tcgetattr(slave) == original
    finally:
        os.close(master)
        os.close(slave)


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP])
def test_shutdown_signals_interrupt_and_restore(signum):
    previous = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP)}
    with pytest.raises(KeyboardInterrupt):
        with ac.shutdown_signals():
            signal.raise_signal(signum)
    assert {s: signal.getsignal(s) for s in previous} == previous


@pytest.mark.parametrize("signum", [signal.SIGHUP, signal.SIGTSTP])
def test_session_signal_during_motion_stops_and_disconnects(rig, monkeypatch, signum):
    c, arm, receiver, *_ = rig
    probe = Arm()
    for device in (probe, arm):
        device.connect = lambda: None
        device.disconnect = lambda device=device: device.calls.append("disconnect")
        device.get_firmware = lambda **kwargs: {"software_version": "S-V1.8-9"}
    receiver.close = lambda: arm.calls.append("receiver_close")
    devices = iter([probe, arm])
    sdk = SimpleNamespace(AgxArmFactory=SimpleNamespace(create_arm=lambda config: next(devices)),
                          create_agx_arm_config=lambda **kwargs: kwargs,
                          resolve_firmware_profile=lambda *args: "v189")
    monkeypatch.setitem(sys.modules, "pyAgxArm", sdk)
    monkeypatch.setitem(sys.modules, "pyAgxArm.api.constants", SimpleNamespace(
        ROBOT_JOINT_LIMIT_PRESET_RAD={"piper_x": dict(enumerate([[-3.14, 3.14]] * 6))}))
    monkeypatch.setattr(ac, "JointReceiver", type("ReceiverFactory", (), {
        "IDS": ac.JointReceiver.IDS, "__new__": lambda cls, channel: receiver}))

    def interrupted_console(controller):
        controller.save_home([0.] * 6)
        controller.command("home")
        signal.raise_signal(signum)

    monkeypatch.setattr(ac, "console", interrupted_console)
    args = SimpleNamespace(channel="can0", home_file=c.home_file, speed=5)
    with ac.shutdown_signals():
        assert ac.run_session(args) == 130
    assert arm.calls[-3:] == ["stop", "receiver_close", "disconnect"]
    assert probe.calls == ["disconnect"]


@pytest.mark.parametrize('mode', [(0, 1), (1, 0)])
def test_wrong_mode_never_sends_target_and_times_out(rig, mode):
    c, arm, receiver, clock, _ = rig
    c.command('jog 6 2')
    for i in range(1, 34):
        clock[0] = 100 + i * .1
        receiver.frames = frames(clock[0])
        receiver.frames[0x2A1] = (bytes([mode[0], 0, mode[1], 0, 0, 0, 0, 0]), clock[0])
        c.tick()
    assert not any(call[0] == 'move' for call in arm.calls if isinstance(call, tuple))
    assert c.locked and arm.calls[-1] == 'stop'


def test_cached_mode_cannot_send_target(rig):
    c, arm, _, _, advance = rig
    c.command('jog 6 2')
    c.tick()
    assert arm.calls == [('speed', 3), ('mode', 'j')]
    for _ in range(8):
        advance()
    assert arm.calls[-1][0] == 'move'
    assert arm.auto_mode
    advance()
    assert sum(isinstance(call, tuple) and call[0] == 'move' for call in arm.calls) == 1


def test_pose_drift_during_mode_wait_cancels_target(rig):
    c, arm, _, _, advance = rig
    c.command('jog 6 2')
    advance(joints=(0, 10000, -10000, 0, 0, 500))
    assert c.locked and arm.calls[-1] == 'stop'
    assert not any(call[0] == 'move' for call in arm.calls if isinstance(call, tuple))


@pytest.mark.parametrize('failure', ['mode', 'fault', 'stale', 'disabled', 'timeout', 'escape'])
def test_failures_after_target_send_stop(rig, failure):
    c, arm, receiver, clock, advance = rig
    c.command('jog 6 2')
    for _ in range(8):
        advance()
    assert arm.calls[-1][0] == 'move'
    if failure in ('mode', 'fault'):
        receiver.frames[0x2A1] = (bytes([0 if failure == 'mode' else 1, 7 if failure == 'fault' else 0, 1, 0, 0, 0, 0, 0]), clock[0])
        c.tick()
    elif failure == 'stale':
        clock[0] += 1
        c.tick()
    elif failure == 'disabled':
        advance(enabled=False)
    elif failure == 'escape':
        c.stop('Esc')
    else:
        advance(31)
    assert c.locked and arm.calls[-1] == 'stop'


@pytest.mark.parametrize("operation", ["jog 6 2", "home", "set-home", "enable"])
def test_not_arrived_flag_does_not_mean_moving(rig, operation):
    c, arm, receiver, clock, _ = rig
    c.save_home(np.zeros(6))
    c.command(operation)
    for _ in range(8):
        clock[0] += .1
        receiver.frames = frames(clock[0])
        receiver.frames[0x2A1] = (bytes([1, 0, 1, 0, 1, 0, 0, 0]), clock[0])
        c.tick()
    assert not c.locked
    if operation in ("home", "jog 6 2"):
        assert c.phase == "等待到位"
        assert c.active is not None  # Flag=1 cannot complete arrival.
        assert sum(isinstance(v, tuple) and v[0] == "move" for v in arm.calls) == 1
    else:
        assert c.active is None


def test_enable_drift_resets_stability_window(rig):
    c, arm, _, _, advance = rig
    c.command("enable")
    for i in range(10):
        advance(joints=(0, 10000, -10000, 0, 0, i * 300))
    assert c.active == "enable"
    for _ in range(8):
        advance(joints=(0, 10000, -10000, 0, 0, 2700))
    assert c.active is None


def test_one_fresh_sample_cannot_complete_stability(rig):
    c, arm, _, clock, advance = rig
    c.command("jog 6 2")
    advance()
    for _ in range(10):
        clock[0] += .01
        c.tick()
    assert c.phase == "等待模式确认"
    assert not any(isinstance(v, tuple) and v[0] == "move" for v in arm.calls)
