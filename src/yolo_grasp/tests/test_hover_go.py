import importlib.util
from pathlib import Path
import sys
import time
import numpy as np
import pytest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hover_red_block as m


def test_local_handeye_accepts_only_the_fixed_pose_d405_02_identity(tmp_path):
    source = Path(__file__).resolve().parents[2] / 'data/handeye/d405_02_fixed_split/candidate_result.json'
    calibration = m.load_local_handeye(source)
    assert calibration['board'] == m.LOCAL_HANDEYE_BOARD
    changed = dict(calibration, session='/tmp/unrelated')
    path = tmp_path / 'wrong.json'
    import json
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='会话身份'):
        m.load_local_handeye(path)


def test_local_tracking_default_ignores_measured_24mm_pose_drift():
    assert m.DEFAULT_TARGET_MOVE_MM == 50.


def test_second_and_later_planning_rounds_get_the_extra_base_offset():
    base = np.array([-43., -28., 0.])
    np.testing.assert_array_equal(m.target_offset_for_plan(base, 1), base)
    np.testing.assert_array_equal(m.target_offset_for_plan(base, 2), [-31., -18., 0.])
    np.testing.assert_array_equal(m.target_offset_for_plan(base, 3), [-31., -18., 0.])
    np.testing.assert_array_equal(m.target_offset_for_plan(base, 4), [-31., -18., 0.])
    np.testing.assert_array_equal(m.target_offset_for_plan(base, 5), [-31., -18., 0.])
    np.testing.assert_array_equal(base, [-43., -28., 0.])


def test_replan_uses_stopped_pose_and_reaches_new_target():
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    mdh = get_mdh('piper')
    fk = lambda q: m.pose_matrix(fk_from_mdh(mdh, list(q)))
    limits = list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values())
    fixed = np.deg2rad(m.TARGET_DEG)
    origin = fk(fixed)
    stopped = origin.copy(); stopped[:3, 3] += [.04, -.025, -.04]
    joints = m.solve_target(stopped, fixed, limits, fk, 90)
    stopped = fk(joints)
    assert np.max(abs(joints-fixed)) > np.deg2rad(1)
    wanted = stopped.copy(); wanted[:3, 3] += [.015, .015, -.010]
    tcp = np.array([0., 0., .080])
    base = wanted[:3, 3]+wanted[:3, :3]@tcp-[0, 0, .035]
    args = SimpleNamespace(clearance_mm=35., max_joint_delta_deg=90., max_travel_mm=200.,
                           target_offset_mm=np.zeros(3), tcp_offset_mm=tcp*1000)
    handeye = np.eye(4)  # Target lies along the flange's positive tool axis.
    report, (times, targets) = m.replan_from_current(
        base, SimpleNamespace(joints=joints), stopped, origin, handeye, limits, fk, args)
    assert not report['issues']
    np.testing.assert_allclose(report['red_block_base_xyz_m'], base, atol=1e-12)
    np.testing.assert_allclose(report['current_joints_rad'], joints)
    assert np.max(abs(targets[0]-joints)) < np.deg2rad(.2)
    d, a = m.pose_error(fk(targets[-1]), wanted)
    assert d < .001 and a < np.deg2rad(.2)
    assert np.max(abs(np.diff(targets, axis=0))/np.diff(times)[:, None]) <= np.deg2rad(5.001)
    far_origin = origin.copy(); far_origin[0, 3] -= 1
    with pytest.raises(ValueError, match='最初的法兰位移包络'):
        m.replan_from_current(base, SimpleNamespace(joints=joints), stopped, far_origin,
                              handeye, limits, fk, args)


@pytest.mark.parametrize('mode', ['recover', 'backlog', 'frozen', 'intermittent'])
def test_feedback_ready_requires_sustained_fresh_updates(monkeypatch, mode):
    from grasp import arm_console as ac
    clock = [100.]
    monkeypatch.setattr(m.time, 'monotonic', lambda: clock[0])
    def sleep(dt):
        clock[0] += dt
    def snapshot():
        elapsed = clock[0] - 100.
        age = .24 if mode == 'backlog' or (mode == 'recover' and elapsed < .15) else .02
        if mode == 'intermittent' and int(round(elapsed/.05)) % 3 == 2:
            age = .24
        stamp = 99.98 if mode == 'frozen' else clock[0] - age
        return {i: (bytes(8), stamp) for i in ac.JointReceiver.IDS}
    receiver = SimpleNamespace(snapshot=snapshot)
    if mode == 'recover':
        m.wait_for_fresh_feedback(receiver, timeout=1., sleep=sleep, now=lambda: clock[0])
        assert clock[0] >= 100.35
    else:
        with pytest.raises(RuntimeError, match='未恢复新鲜'):
            m.wait_for_fresh_feedback(receiver, timeout=.6, sleep=sleep, now=lambda: clock[0])


def test_reject_diagnostic_issues():
    with pytest.raises(ValueError, match='坐标检查未通过'):
        m.make_path({'issues': ['当前 CAN 与 piper FK 不一致']}, [], None)


@pytest.mark.parametrize('value', ['201', '0', '-1', 'nan'])
def test_entry_point_rejects_out_of_range_travel(monkeypatch, value):
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    with pytest.raises(SystemExit) as error:
        m.main(['--max-travel-mm', value])
    assert error.value.code == 2


@pytest.mark.parametrize('value', ['0.1', '11', 'nan'])
def test_entry_point_rejects_out_of_range_hold_tolerance(monkeypatch, value):
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    with pytest.raises(SystemExit) as error:
        m.main(['--hold-tolerance-deg', value])
    assert error.value.code == 2


@pytest.mark.parametrize('value', ['51', 'nan'])
def test_entry_point_rejects_out_of_range_target_offset(monkeypatch, value):
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    with pytest.raises(SystemExit) as error:
        m.main(['--target-offset-mm', value, '0', '0'])
    assert error.value.code == 2


@pytest.mark.parametrize('argv', [['--grasp', '--grip-force-n', '0'],
                                  ['--grasp', '--grip-force-n', '51'],
                                  ['--grasp', '--grip-force-n', '3', '--lift-mm', '0'],
                                  ['--grasp', '--grip-force-n', '3', '--lift-mm', '101'],
                                  ['--grasp', '--grip-force-n', '1', '--clearance-mm', '35', '--lift-to-mm', '30'],
                                  ['--grasp', '--grip-force-n', '1', '--lift-to-mm', '201'],
                                  ['--grasp', '--grip-force-n', '1', '--lift-mm', '20', '--lift-to-mm', '80'],
                                  ['--grasp', '--grip-force-n', '1', '--grip-step-mm', '6'],
                                  ['--grasp', '--grip-force-n', '1', '--grip-step-ms', '20']])
def test_grasp_requires_a_bounded_force_and_lift(monkeypatch, argv):
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    with pytest.raises(SystemExit) as error:
        m.main(argv)
    assert error.value.code == 2


class GraspRig:
    """Minimal stand-ins for the grasp/lift phase."""
    def __init__(self, width_after_lift=.025):
        self.controller = SimpleNamespace(state=lambda: SimpleNamespace(joints=np.zeros(6)),
                                          locked=False)
        self.closed = []
        self.executed = []
        self.width_after_lift = width_after_lift
        self.arm = SimpleNamespace(
            OPTIONS=SimpleNamespace(EFFECTOR=SimpleNamespace(AGX_GRIPPER='agx')),
            init_effector=lambda name: SimpleNamespace(
                move_gripper_m=lambda value, force: self.closed.append((value, force))))
        flange = np.eye(4); flange[:3, 3] = [.3, 0, .2]
        self.flange = flange
        self.receiver = SimpleNamespace()

    def snapshot(self, receiver, now): return None, self.flange, None


def test_grasp_closes_with_the_given_force_then_lifts(monkeypatch):
    rig = GraspRig()
    args = SimpleNamespace(channel='can1', grip_force_n=3., lift_mm=30., lift_to_mm=None,
                           grip_step_mm=1., grip_step_ms=200., max_joint_delta_deg=90, hold_tolerance_deg=.8)
    events, lifted = [], []
    monkeypatch.setattr(m, 'wait_for_gripper_feedback', lambda feedback, **k: feedback.latest())
    monkeypatch.setattr(m, 'GripperFeedback', lambda channel: SimpleNamespace(
        start=lambda: None, latest=lambda: dict(width_m=.025, force_n=3., active=True, stamp=0.), close=lambda: None))
    def fake_close(command, read, force_n, step_m=None, step_s=None, event=None):
        command(.0245)
        if event is not None:
            event(dict(event='gripper_step', target_mm=24.5, width_mm=25., force_n=1., step=1))
        return dict(gripped=True, width_m=.025, force_n=3., steps=7, target_mm=24.5)
    monkeypatch.setattr(m, 'close_until_blocked', fake_close)
    monkeypatch.setattr(m, 'make_path', lambda report, limits, fk, deg: np.zeros((3, 6)))
    monkeypatch.setattr(m, 'timed_path', lambda path, speed_deg_s: (np.array([0., 1.]), np.zeros((2, 6))))
    monkeypatch.setattr(m, 'wait_for_fresh_feedback', lambda receiver, **k: None)
    monkeypatch.setattr(m, 'wait_for_fresh_feedback', lambda receiver, **k: None)
    def lift_flange(*a, **k):
        lifted.append(k)
        rig.flange[:3, 3] = rig.flange[:3, 3] + np.array([0., 0., .03])
    monkeypatch.setattr(m, 'execute_path', lift_flange)

    m.grasp_and_lift(rig.controller, rig.receiver, rig.arm, args, lambda: None,
                     events.append, [[-1, 1]]*6, object(), rig.snapshot)

    assert rig.closed == [(.0245, 3.)], 'the closing step must use --grip-force-n'
    assert lifted and lifted[0]['hold_tolerance_deg'] == .8
    names = [e['event'] for e in events]
    assert names == ['gripper_close_requested', 'gripper_feedback_ready', 'gripper_step',
                     'gripper_closed', 'lift_plan', 'lift_reached']
    assert events[-1]['lift_mm'] == 30. and events[-1]['width_mm'] == pytest.approx(25.)


def test_lift_to_mm_targets_the_height_above_the_block(monkeypatch):
    """--clearance-mm 35 with --lift-to-mm 80 lifts exactly 45 mm."""
    rig = GraspRig()
    args = SimpleNamespace(channel='can1', grip_force_n=1., clearance_mm=35., lift_mm=None,
                           lift_to_mm=80., grip_step_mm=1., grip_step_ms=200.,
                           max_joint_delta_deg=90, hold_tolerance_deg=.8)
    events = []
    monkeypatch.setattr(m, 'wait_for_gripper_feedback', lambda feedback, **k: feedback.latest())
    monkeypatch.setattr(m, 'GripperFeedback', lambda channel: SimpleNamespace(
        start=lambda: None, latest=lambda: dict(width_m=.025, force_n=1., active=True, stamp=0.),
        close=lambda: None))
    def fake_close(command, read, force_n, step_m=None, step_s=None, event=None):
        return dict(gripped=True, width_m=.025, force_n=1., steps=7, target_mm=24.5)
    monkeypatch.setattr(m, 'close_until_blocked', fake_close)
    monkeypatch.setattr(m, 'make_path', lambda report, limits, fk, deg: np.zeros((3, 6)))
    monkeypatch.setattr(m, 'timed_path', lambda path, speed_deg_s: (np.array([0., 1.]), np.zeros((2, 6))))
    monkeypatch.setattr(m, 'wait_for_fresh_feedback', lambda receiver, **k: None)
    def lift_flange(*a, **k):
        rig.flange[:3, 3] = rig.flange[:3, 3] + np.array([0., 0., .045])
    monkeypatch.setattr(m, 'execute_path', lift_flange)
    m.grasp_and_lift(rig.controller, rig.receiver, rig.arm, args, lambda: None,
                     events.append, [[-1, 1]]*6, object(), rig.snapshot)
    plan = [e for e in events if e['event'] == 'lift_plan'][0]
    assert plan['lift_mm'] == pytest.approx(45.) and plan['lift_to_mm'] == 80.
    assert events[-1]['lift_mm'] == pytest.approx(45.)


def test_grasp_refuses_to_lift_when_the_block_slips(monkeypatch):
    rig = GraspRig()
    args = SimpleNamespace(channel='can1', grip_force_n=3., lift_mm=30., lift_to_mm=None,
                           grip_step_mm=1., grip_step_ms=200., max_joint_delta_deg=90, hold_tolerance_deg=.8)
    widths = iter([dict(width_m=.025, force_n=3., active=True, stamp=0.),
                   dict(width_m=.021, force_n=1., active=True, stamp=0.)])
    monkeypatch.setattr(m, 'wait_for_gripper_feedback', lambda feedback, **k: next(widths))
    monkeypatch.setattr(m, 'GripperFeedback', lambda channel: SimpleNamespace(
        start=lambda: None, latest=lambda: next(widths), close=lambda: None))
    def fake_close(command, read, force_n, step_m=None, step_s=None, event=None):
        return dict(gripped=True, width_m=.025, force_n=3., steps=7, target_mm=24.5)
    monkeypatch.setattr(m, 'close_until_blocked', fake_close)
    monkeypatch.setattr(m, 'make_path', lambda report, limits, fk, deg: np.zeros((3, 6)))
    monkeypatch.setattr(m, 'timed_path', lambda path, speed_deg_s: (np.array([0., 1.]), np.zeros((2, 6))))
    monkeypatch.setattr(m, 'wait_for_fresh_feedback', lambda receiver, **k: None)
    monkeypatch.setattr(m, 'execute_path', lambda *a, **k: rig.flange.__setitem__(
        (slice(3), 3), rig.flange[:3, 3] + np.array([0., 0., .03])))
    with pytest.raises(ValueError, match='滑脱'):
        m.grasp_and_lift(rig.controller, rig.receiver, rig.arm, args, lambda: None,
                         [].append, [[-1, 1]]*6, object(), rig.snapshot)


def test_parked_flange_tolerance_covers_the_documented_snap():
    metres, angle = m.parked_flange_tolerance(.8)
    assert metres == pytest.approx(.004) and np.rad2deg(angle) == pytest.approx(.8)
    metres, angle = m.parked_flange_tolerance(.2)
    assert metres == pytest.approx(.002) and np.rad2deg(angle) == pytest.approx(.2)


def test_path_rejects_accumulated_deviation(monkeypatch):
    q = np.deg2rad(m.TARGET_DEG)
    report = dict(issues=[], current_CAN_flange_m_rad=[0]*6,
                  hypothetical_target_flange_m_rad=[0]*6, current_joints_rad=q.tolist())
    monkeypatch.setattr(m, 'solve_target',
                        lambda pose, seed, limits, fk, max_delta=None: seed + np.deg2rad(.2))
    with pytest.raises(ValueError, match='累计'):
        m.make_path(report, [], None)


class Controller:
    locked = False
    def tick(self): pass


class Terminal:
    """fd stand-in for the shared word reader (grasp.terminal_input)."""
    def __init__(self, typed, ready=lambda: True, empty_is_eof=False):
        self.buffer = bytearray(typed)
        self.ready = ready
        self.empty_is_eof = empty_is_eof

    def select(self, rlist, wlist, xlist, timeout=0):
        if self.buffer or self.empty_is_eof:
            return ([rlist[0]], [], []) if self.ready() else ([], [], [])
        return ([], [], [])

    def read(self, fd, count):
        chunk = self.buffer[:count]
        del self.buffer[:count]
        return bytes(chunk)


def fake_terminal(monkeypatch, typed, ready=lambda: True, empty_is_eof=False):
    from grasp import terminal_input
    terminal = Terminal(typed, ready, empty_is_eof)
    monkeypatch.setattr(terminal_input.select, 'select', terminal.select)
    monkeypatch.setattr(terminal_input.os, 'read', terminal.read)
    return terminal


def test_only_exact_go_releases_motion(monkeypatch):
    terminal = fake_terminal(monkeypatch, b'next\ngo\n')
    monkeypatch.setattr(m.time, 'monotonic', lambda: 10)
    m.wait_go(Controller(), 1, 10)
    assert not terminal.buffer


@pytest.mark.parametrize('backspace', [b'\x7f', b'\x08'])
def test_go_backspace_erases_display_and_corrects_command(monkeypatch, capsys, backspace):
    fake_terminal(monkeypatch, backspace + b'gx' + backspace + b'o\n')
    monkeypatch.setattr(m.time, 'monotonic', lambda: 10)
    m.wait_go(Controller(), 1, 10)
    output = capsys.readouterr().out
    assert 'gx\b \bo\n' in output
    assert output.count('\b \b') == 1


def test_go_accepts_the_delete_key_instead_of_stopping(monkeypatch):
    """Delete sends `ESC [ 3 ~`; it must edit the word, not request a stop."""
    fake_terminal(monkeypatch, b'goo\x1b[3~\n')
    monkeypatch.setattr(m.time, 'monotonic', lambda: 10)
    m.wait_go(Controller(), 1, 10)


def test_stale_observation_never_accepts_go(monkeypatch):
    monkeypatch.setattr(m.time, 'monotonic', lambda: 41)
    with pytest.raises(TimeoutError):
        m.wait_go(Controller(), 1, 10)


def test_stop_before_go(monkeypatch):
    monkeypatch.setattr(m.time, 'monotonic', lambda: 10)
    fake_terminal(monkeypatch, b' ')
    with pytest.raises(KeyboardInterrupt):
        m.wait_go(Controller(), 1, 10)


class HomeController:
    """Minimal stand-in for the guarded controller used by go_home()."""
    def __init__(self, start, home):
        self.joints = np.asarray(start, float)
        self.active = None
        self.guard_start = None
        self.guard_goal = None
        self.experiment_target = None
        self.commands = []
    def state(self):
        return SimpleNamespace(joints=self.joints.copy())
    def command(self, name):
        self.commands.append(name)
        if name == 'home':
            self.joints = np.asarray(self.experiment_target, float)
            self.active = 'home'


def test_home_returns_to_the_saved_initial_pose():
    start = np.deg2rad(m.TARGET_DEG)
    home = start + np.deg2rad([1.0, -2.0, .5, .25, -1.5, .75])
    controller = HomeController(start, home)
    ticks = []
    def tick():
        ticks.append(1)
        if controller.active == 'home':
            controller.active = None
    result = m.go_home(controller, tick, home_joints=home)
    np.testing.assert_allclose(result, home)
    assert controller.commands == ['home']
    np.testing.assert_allclose(controller.guard_goal, home)
    np.testing.assert_allclose(controller.guard_start, start)
    assert ticks, 'the caller tick must keep running during the move'


def test_home_refuses_to_claim_success_when_the_arm_stops_short():
    start = np.deg2rad(m.TARGET_DEG)
    home = start + np.deg2rad([1.0, -2.0, .5, .25, -1.5, .75])
    controller = HomeController(start, home)
    def tick():
        if controller.active == 'home':
            controller.joints = home + np.deg2rad([0, 0, 0, 0, 0, .9])
            controller.active = None
    with pytest.raises(ValueError, match='回初始位置误差'):
        m.go_home(controller, tick, home_joints=home)


class TwoStepController:
    """Records the joint target of every commanded leg."""
    def __init__(self, start):
        self.joints = np.asarray(start, float)
        self.active = None
        self.experiment_target = None
        self.commands = []
        self.reached = []
    def state(self):
        return SimpleNamespace(joints=self.joints.copy())
    def command(self, name):
        self.commands.append(name)
        self.active = name
    def tick(self):
        if self.active is not None:
            self.joints = np.asarray(self.experiment_target, float)
            self.reached.append(self.joints.copy())
            self.active = None


def test_home_visits_the_fixed_grasp_pose_before_the_initial_pose():
    start = np.deg2rad(m.TARGET_DEG) + np.deg2rad([2, -3, 1, .5, -1, .25])
    home = np.deg2rad(m.TARGET_DEG) + np.deg2rad([1.0, -2.0, .5, .25, -1.5, .75])
    controller = TwoStepController(start)
    result = m.go_home_via_fixed_pose(controller, controller.tick, home_joints=home)
    np.testing.assert_allclose(result, home)
    assert controller.commands == ['home', 'home']
    assert len(controller.reached) == 2
    np.testing.assert_allclose(controller.reached[0], np.deg2rad(m.TARGET_DEG))
    np.testing.assert_allclose(controller.reached[1], home)


def test_home_skips_the_first_leg_when_already_at_the_fixed_pose():
    start = np.deg2rad(m.TARGET_DEG)
    home = start + np.deg2rad([1.0, -2.0, .5, .25, -1.5, .75])
    controller = TwoStepController(start)
    m.go_home_via_fixed_pose(controller, controller.tick, home_joints=home)
    assert controller.commands == ['home']
    np.testing.assert_allclose(controller.reached[0], home)


def test_handeye_audit_indices_stay_in_safe_first_sixty_percent():
    np.testing.assert_equal(m.handeye_audit_indices(151, [0, .15, .30, .45]), [0, 22, 45, 68])
    with pytest.raises(ValueError, match='不超过0.6'):
        m.handeye_audit_indices(151, [0, .5, .7])
    with pytest.raises(ValueError, match='严格递增'):
        m.handeye_audit_indices(151, [0, .3, .3])


class HoldController:
    active = None
    locked = False
    def __init__(self):
        self.ticks = 0
    def state(self): return SimpleNamespace(joints=np.zeros(6))
    def tick(self): self.ticks += 1
    def command(self, *args): raise AssertionError('hold must not issue commands')
    def stop(self, *args): raise AssertionError('normal hold must not stop')


def test_completed_motion_holds_without_deadline_or_commands(monkeypatch):
    c = HoldController()
    events = []
    # The word only becomes readable after the hold has been monitoring a while.
    fake_terminal(monkeypatch, b'quit\n', ready=lambda: c.ticks > 600)
    monkeypatch.setattr(m.time, 'monotonic', lambda: c.ticks*10.)
    preview = SimpleNamespace(pump=lambda **kw: None)
    assert m.hold_after_motion(c, 1, preview, events.append) == 0
    assert c.ticks > 600
    assert events[0]['event'] == 'hold_started'
    assert events[-1]['event'] == 'hold_exit'


def test_recoverable_failure_holds_for_home_or_quit_without_stopping(monkeypatch, capsys):
    c = HoldController(); events = []
    fake_terminal(monkeypatch, b'quit\n')
    error = RuntimeError('停稳后未检测到红块')
    assert m.hold_after_motion(c, 1, SimpleNamespace(pump=lambda **kw: None),
                               events.append, failure=error) == 0
    assert events[0]['failure'] == str(error)
    assert not c.locked
    assert '本次抓取已安全停止' in capsys.readouterr().out


def test_hold_preview_failure_does_not_stop_robot_monitor(monkeypatch):
    c = HoldController(); events = []; pumps = []
    def pump(**kwargs):
        pumps.append(1)
        raise RuntimeError('camera disconnected')
    fake_terminal(monkeypatch, b'quit\n')
    assert m.hold_after_motion(c, 1, SimpleNamespace(pump=pump), events.append) == 0
    assert c.ticks >= 1 and len(pumps) == 1
    assert any(e['event'] == 'hold_preview_unavailable' for e in events)


@pytest.mark.parametrize('key', [b' ', b'\x1b', b'\x03'])
def test_hold_preserves_manual_stop(monkeypatch, key):
    fake_terminal(monkeypatch, key)
    with pytest.raises(KeyboardInterrupt):
        m.hold_after_motion(HoldController(), 1, SimpleNamespace(pump=lambda **kw: None), lambda e: None)


@pytest.mark.parametrize('key', [b'', b'\x04'])
def test_hold_eof_is_clean_exit(monkeypatch, key):
    fake_terminal(monkeypatch, key, empty_is_eof=True)
    assert m.hold_after_motion(HoldController(), 1, SimpleNamespace(pump=lambda **kw: None), lambda e: None) == 0


def test_hold_robot_fault_is_not_swallowed_as_preview_error():
    c = HoldController()
    def tick(): raise RuntimeError('CAN 反馈过期')
    c.tick = tick
    with pytest.raises(RuntimeError, match='CAN 反馈过期'):
        m.hold_after_motion(c, 1, SimpleNamespace(pump=lambda **kw: None), lambda e: None)


def test_hold_window_stop_is_still_manual_emergency():
    def pump(**kwargs): raise KeyboardInterrupt('相机窗口停止键')
    with pytest.raises(KeyboardInterrupt):
        m.hold_after_motion(HoldController(), 1, SimpleNamespace(pump=pump), lambda e: None)


def test_hold_place_continues_only_with_the_flag(monkeypatch):
    """`place` hands the arm (still holding the block) to the placement leg."""
    events = []
    fake_terminal(monkeypatch, b'place\n')
    c = HoldController()
    assert m.hold_after_motion(c, 1, SimpleNamespace(pump=lambda **kw: None), events.append,
                               allow_place=True) == m.PLACE_CONTINUE
    assert events[-1] == dict(event='hold_exit', reason='place_continue', enabled_hold=True)
    assert not c.locked, '继续放置时不能急停或失能'


def test_hold_place_without_the_flag_exits_like_quit(monkeypatch, capsys):
    events = []
    fake_terminal(monkeypatch, b'place\n')
    assert m.hold_after_motion(HoldController(), 1, SimpleNamespace(pump=lambda **kw: None),
                               events.append) == 0
    assert events[-1]['reason'] == 'place_unavailable'
    assert '只在组合入口' in capsys.readouterr().out


def test_grasp_entry_forwards_continue_to_place(monkeypatch):
    """The combined entry's flag reaches the hold without touching motion."""
    seen = {}
    def fake_hold(controller, fd, preview, event, allow_place=False):
        seen['allow_place'] = allow_place
        return m.PLACE_CONTINUE
    monkeypatch.setattr(m, 'hold_after_motion', fake_hold)
    assert m.PLACE_CONTINUE == 10 and fake_hold(None, None, None, lambda e: None, allow_place=True) == 10
    assert seen['allow_place'] is True


def test_strike_trace_dump_moves_the_disk_work_off_the_control_tick(tmp_path):
    """A slow trace dump must not stall the caller (2026-09-18 abort)."""
    class SlowTrace:
        def dump(self):
            time.sleep(.3)
            return [dict(can_id='0x2a5', data_hex='00')]

    path = tmp_path/'guard_strike_01.jsonl'
    events, threads = [], []
    started = time.monotonic()
    m.dump_trace_async(SlowTrace(), path, events.append, threads, path.name)
    assert time.monotonic()-started < .05
    assert not path.exists()
    for thread in threads:
        thread.join(timeout=2)
    assert path.exists() and '0x2a5' in path.read_text()
    assert events == [dict(event='guard_strike_dump', trace_file=path.name, trace_rows=1)]


def test_strike_trace_dump_reports_listener_failure_without_raising(tmp_path):
    class DeadTrace:
        def dump(self):
            raise RuntimeError('双通道 CAN 记录线程失败')

    events, threads = [], []
    m.dump_trace_async(DeadTrace(), tmp_path/'strike.jsonl', events.append, threads, 'strike.jsonl')
    for thread in threads:
        thread.join(timeout=2)
    assert events == [dict(event='can_trace_failed', error='双通道 CAN 记录线程失败')]
