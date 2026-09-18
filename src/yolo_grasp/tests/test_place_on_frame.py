import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import place_on_frame as m


class Frame:
    def __init__(self, depth=0.2, shape=(240, 320)):
        self.depth_m = np.full(shape, depth)


def boxed(xyxy=(100, 80, 200, 160)):
    array_like = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array(xyxy, float)))
    return SimpleNamespace(boxes=[SimpleNamespace(xyxy=[array_like])], reason=None)


def test_black_frame_point_uses_the_box_centre_depth():
    def locate(frame, box, min_depth, max_depth):
        assert list(box) == [100, 80, 200, 160]
        return dict(valid=True, xyz_camera_m=[0, 0, .2])
    point = m.black_frame_point(boxed(), Frame(), locate)
    assert point['valid'] and point['xyz_camera_m'] == [0, 0, .2]


class LivePreview:
    def __init__(self, sample):
        self._sample = sample
    def sample(self):
        return self._sample


def live_sample(stamp, box=boxed()):
    return (stamp, Frame(), box, None)


def test_live_watch_records_detections_without_touching_the_plan(capsys):
    events = []
    times = iter([1., 2., 3.])
    watch = m.LiveFrameWatch([.40, .10, .20], np.eye(4),
                             lambda frame, box, **kw: dict(valid=True, xyz_camera_m=[.40, .10, .20]),
                             events.append, period=0., now=lambda: next(times))
    preview = LivePreview(live_sample(1.))
    for _ in range(2):
        watch.update(preview, lambda: (None, np.eye(4), None))
    # The same camera stamp is only counted once (no duplicate events per frame).
    preview = LivePreview(live_sample(2.))
    watch.update(preview, lambda: (None, np.eye(4), None))
    assert [event['event'] for event in events] == ['frame_live', 'frame_live']
    assert events[0]['base_xyz_m'] == [.40, .10, .20]
    assert events[0]['delta_from_plan_mm'] == [0., 0., 0.]
    assert watch.seen == 2 and watch.lost == 0
    assert '有效 2 帧' in watch.summary()
    # The terminal must not be flooded while the arm moves.
    assert capsys.readouterr().out == ''


def test_live_watch_reports_a_lost_frame_once_and_keeps_running(capsys):
    events = []
    times = iter([1., 2., 3.])
    watch = m.LiveFrameWatch([.40, .10, .20], np.eye(4),
                             lambda frame, box, **kw: dict(valid=False, reason='black_frame_count'),
                             events.append, period=0., now=lambda: next(times))
    preview = LivePreview(live_sample(1.))
    watch.update(preview, lambda: (None, np.eye(4), None))
    preview = LivePreview(live_sample(2.))
    watch.update(preview, lambda: (None, np.eye(4), None))
    assert [event['event'] for event in events] == ['frame_live_lost']
    assert events[0]['reason'] == 'black_frame_count'
    assert watch.lost == 2 and watch.seen == 0
    assert '未识别到黑框' in watch.summary()
    assert capsys.readouterr().out == ''


def test_pre_shift_moves_the_tcp_in_the_base_frame():
    flange = np.eye(4)
    flange[:3, :3] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    flange[:3, 3] = [.3, -.05, .2]
    target = m.shifted_flange(flange, [0., -80., 0.])
    np.testing.assert_allclose(target[:3, 3], [.3, -.13, .2])
    np.testing.assert_allclose(target[:3, :3], flange[:3, :3])
    with pytest.raises(ValueError, match='预移动量'):
        m.shifted_flange(flange, [0., 0., 300.])


def test_release_steps_open_from_the_measured_width():
    """Opening walks the width up (closing is the leg that has to stall on a block)."""
    steps = m.open_steps(.033, .100, .001)
    assert len(steps) == 67 and steps[0] == pytest.approx(.034) and steps[-1] == pytest.approx(.100)
    assert m.open_steps(.100, .100, .002) == [.100]          # 已经在目标开度
    assert m.open_steps(.050, .100, .020) == pytest.approx([.070, .090, .100])
    with pytest.raises(ValueError, match='张爪'):
        m.open_steps(.050, .100, .0)
    with pytest.raises(ValueError, match='张爪'):
        m.open_steps(.050, 1.5, .002)


def test_defaults_detect_at_the_fixed_pose_without_the_shift_step():
    """Detection now runs at the fixed grasp pose; `shift` is opt-in again."""
    args = m.build_parser().parse_args([])
    assert list(args.pre_shift_mm) == [0., 0., 0.]
    assert args.clearance_mm == 55.
    assert list(args.target_offset_mm) == [5., 5., 0.]
    assert args.max_j5_deg == 60. and args.max_tilt_deg == 25.
    # 2026-09-14 从实测 4/3 档小幅提速；守卫阈值保持不变。
    assert (args.path_speed_deg_s, args.final_speed_deg_s) == (7., 5.)
    assert m.PLACE_SPEED_PERCENT == 5
    assert (args.follow_error_deg, args.motion_envelope_deg) == (1.2, .8)


def test_j5_cap_is_planning_only_and_leaves_the_controller_table_alone():
    """Regression for the 2026-09-14 `home` abort: 第 5 轴 60.349° 关节越界.

    The controller validates raw feedback with no tolerance, so feeding the capped
    table to it turns a documented 0.3-0.4° joint-frame glitch at the cap into an
    immediate stop. Planning uses the capped copy, the controller keeps ±89°.
    """
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    limits = list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values())
    capped = m.cap_plan_limits(limits, 60.)
    assert np.allclose(np.rad2deg(capped[4]), [-60., 60.])
    assert np.allclose(np.rad2deg(limits[4]), [-89., 89.])      # 原表未被改动
    assert capped is not limits and list(capped[0]) == list(limits[0])
    glitch = np.zeros(6); glitch[4] = np.deg2rad(60.349)
    bounds = np.asarray(limits)
    assert ((glitch >= bounds[:, 0]) & (glitch <= bounds[:, 1])).all()
    assert np.asarray(capped)[4, 1] < glitch[4]                  # 规划表里才算越界


def test_one_shot_target_uses_the_fixed_orientation():
    """Flange z is tilted, so the TCP offset must be rotated, not added raw."""
    flange = np.eye(4)
    flange[:3, :3] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    flange[:3, 3] = [.2, 0, .35]
    offset = np.array([0., 0., .08])
    # Absolute target: the flange moves back along its own z so the TCP lands there.
    target, tcp = m.tcp_target_pose(flange, offset, base_mm=[370.2, 126.5, 213.8])
    np.testing.assert_allclose(tcp, [.3702, .1265, .2138])
    np.testing.assert_allclose(target[:3, 3], [.2902, .1265, .2138])
    np.testing.assert_allclose(target[:3, :3], flange[:3, :3])
    # Relative target: measured delta from the fixed pose (2026-09-11 capture).
    delta = [110.6, 152.9, -92.2]
    target2, tcp2 = m.tcp_target_pose(flange, offset, delta_mm=delta)
    now = flange[:3, 3] + flange[:3, :3] @ offset
    np.testing.assert_allclose(tcp2, now + np.array(delta)/1000)
    with pytest.raises(ValueError, match='只能给出'):
        m.tcp_target_pose(flange, offset)
    with pytest.raises(ValueError, match='只能给出'):
        m.tcp_target_pose(flange, offset, base_mm=[0, 0, 0], delta_mm=[0, 0, 0])


def test_line_path_slerps_the_orientation():
    """A big reorientation must be spread over the path, not demanded in one step."""
    from grasp.geometry import pose_matrix, pose_error
    from scipy.spatial.transform import Rotation
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    q0 = np.deg2rad(np.array(m.TARGET_DEG))
    start = fk(q0)
    target = start.copy()
    target[:3, :3] = Rotation.from_euler('xyz', np.deg2rad([0., 180., 0.])).as_matrix()
    target[:3, 3] = start[:3, 3] + [.03, .02, -.02]
    path = m.make_line_path(start, target, q0, [[-3.14, 3.14]]*6, fk, 170.)
    steps = np.rad2deg(np.max(np.abs(np.diff(path, axis=0))))
    assert len(path) >= 151 and steps.max() <= .5
    distance, angle = pose_error(fk(path[-1]), target)
    assert distance < 1e-3 and angle < np.deg2rad(.2)


def test_tool_down_is_only_asked_for_where_the_wrist_can_hold_it():
    """The descent turns the tool axis down on the way, it is not demanded above.

    Regression for 2026-09-14 (`runs/place_frame_20260914_105052`): the old plan
    completed the attitude turn on the level approach, so the IK was asked for a
    tool-down pose 113 mm above the target, came back with a 0.31° residual (J5
    clamped at its ±89° limit) and the session aborted. The turn now happens
    during the descent and the last part is a straight vertical drop with the tool
    axis already down.
    """
    from grasp.geometry import pose_matrix
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    q0 = np.deg2rad(m.TARGET_DEG)
    start = fk(q0)
    offset = np.array([0., 0., .08])
    target, tcp = m.tcp_target_pose(start, offset, base_mm=[370.2, 126.5, 193.8],
                                   rpy_deg=[0., 180., 0.])
    approach, descent = m.vertical_paths(start, target, q0, [[-3.14, 3.14]]*6,
                                        fk, 120., offset, 400.)
    path = approach + descent[1:]
    poses = [fk(q) for q in path]
    tcps = np.array([p[:3, 3] + p[:3, :3] @ offset for p in poses])
    tool_down = np.array([np.linalg.norm(p[:3, 2] - [0., 0., -1.]) < .005 for p in poses])
    # The level approach keeps the current attitude and the TCP height while it
    # travels; that is the part the joint limits cannot do with the tool down.
    np.testing.assert_allclose(poses[0][:3, :3], start[:3, :3])
    assert np.ptp(tcps[:len(approach), 2]) < 1e-9
    assert np.max(abs(tcps[:len(approach), :2] - tcps[0, :2])) > .05
    assert not tool_down[0] and tool_down[-1]
    # Everything from the last tilted waypoint down is a straight vertical drop
    # with the tool already down, and it reaches the requested TCP exactly.
    first = int(np.max(np.flatnonzero(~tool_down))) + 1
    drop = tcps[first:]
    assert tool_down[first:].all()
    assert np.max(abs(drop[:, :2] - tcp[:2])) < .001
    assert np.max(np.diff(drop[:, 2])) < .0001
    np.testing.assert_allclose(tcps[-1], tcp, atol=.001)
    assert tcps[len(approach), 2] >= tcp[2] + .069
    assert np.max(np.abs(np.diff(np.array(path), axis=0))) <= np.deg2rad(.5)
    with pytest.raises(ValueError, match='工具轴'):
        m.vertical_paths(start, start, q0, [], fk, 120., offset, 400.)
    with pytest.raises(ValueError, match='位移'):
        m.vertical_paths(start, target, q0, [[-3.14, 3.14]]*6, fk, 120., offset, 10.)


def test_recorded_20260914_pose_no_longer_stops_on_the_ik_residual():
    """The exact pose that stopped the 2026-09-14 run now plans to the end."""
    from grasp.geometry import pose_matrix, pose_error, transform
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    limits = list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values())
    joints = np.array([0.5073148536771918, 1.0435672663524496, -0.9559691911948541,
                       -0.6585650866550203, 0.8646710180230308, 0.811665368639963])
    flange = transform([[-0.6039202024427244, 0.07803459496949203, 0.7932156018822916,
                         0.19630799999999998],
                        [0.008011170776195412, 0.9957397919793329, -0.09185906494053417,
                         0.060638],
                        [-0.7970045233399452, -0.04912095944606509, -0.601972525219187,
                         0.35359199999999996],
                        [0., 0., 0., 1.]])
    target = pose_matrix([0.3815992740626556, 0.12706031224568332, 0.27191901441125893,
                          np.pi, 0., np.pi])
    offset = np.array([0., 0., .08])
    end_tcp = target[:3, 3] + target[:3, :3] @ offset
    # The wrist only takes the tool-down attitude this close to the target.
    reach = m.tool_down_reach(target, end_tcp, .1135, joints, limits, fk, 120., offset)
    assert .005 < reach < .030
    # The old plan finished the attitude turn on the level approach, i.e. it asked
    # for a tool-down pose 113 mm above the target; that is the 0.31° stop.
    overhead = target.copy()
    overhead[:3, 3] = np.r_[end_tcp[:2], end_tcp[2] + .1135] - target[:3, :3] @ offset
    with pytest.raises(ValueError, match='残差'):
        m.make_line_path(flange, overhead, joints, limits, fk, 120., tcp_offset_m=offset)
    approach, descent = m.vertical_paths(flange, target, joints, limits, fk, 120.,
                                         offset, 400.)
    path = np.array(approach + descent[1:])
    assert np.max(np.abs(np.diff(path, axis=0))) <= np.deg2rad(.5)
    assert np.max(np.abs(path - joints)) <= np.deg2rad(120.)
    distance, angle = pose_error(fk(path[-1]), target)
    assert distance < 1e-6 and angle < np.deg2rad(.01)


def test_j5_cap_tilts_the_target_instead_of_pushing_into_the_bound():
    """The 13:40 run's frame: J5 cannot hold 0° here, so aim for the least tilt.

    J5 stopped at 67.0-67.7° with the servo at 12 A (2026-09-14, three runs); a
    tool-down attitude at this frame needs 86.9°, so with the cap at 60° the plan
    must settle for ~21° of tilt and keep every commanded J5 inside the cap.
    """
    from grasp.geometry import pose_matrix, pose_error, transform
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    base = np.array([0.3963524055059717, 0.1345340754511361, 0.16128557364572027])
    offset = np.array([0., 0., .08])
    joints = np.array([0.5080478919630294, 1.0431658406244908, -0.9565800564330521,
                       -0.6580414878794221, 0.8653167898462686, 0.8120667943679216])
    flange = transform([[-0.6037568025305322, 0.07794870968233368, 0.7933484241220673,
                         0.196159],
                        [0.008757313197419292, 0.995796354856896, -0.09117526594028313,
                         0.060729],
                        [-0.7971204632072061, -0.048100086409260286, -0.6019014444439816,
                         0.353804],
                        [0., 0., 0., 1.]])
    handeye = transform([[0.030709463333916335, 0.874745211299325, 0.4836090819770054,
                          -0.0719410711858362],
                         [-0.999239663546388, 0.01523997655857701, 0.03588645859088458,
                          0.011369355362950598],
                         [0.02402131673002336, -0.48434343024696136, 0.8745481221288871,
                          0.04750018420036297],
                         [0., 0., 0., 1.]])
    camera_point = (np.linalg.inv(flange @ handeye) @ np.r_[base, 1.])[:3]
    limits = list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values())
    limits[4] = [-np.deg2rad(60.), np.deg2rad(60.)]
    report = m.plan_report(joints, flange, camera_point, handeye, limits, fk, .030, 120.,
                           400., np.zeros(3), offset, 60., 25.)
    assert report['issues'] == []
    assert 19. <= report['tool_tilt_deg'] <= 23.
    assert abs(report['IK_joints_deg'][4]) <= 60.
    target = pose_matrix(report['hypothetical_target_flange_m_rad'])
    approach, descent = m.vertical_paths(flange, target, joints, limits, fk, 120., offset,
                                        400., max_tilt_deg=25.)
    path = np.array(approach + descent[1:])
    assert np.rad2deg(np.max(np.abs(path[:, 4]))) <= 60. + 1e-9
    assert np.max(np.abs(np.diff(path, axis=0))) <= np.deg2rad(.5)
    distance, angle = pose_error(fk(path[-1]), target)
    assert distance < 1e-6 and angle < np.deg2rad(.01)
    # The tail is a straight vertical drop with the attitude held (tilt and XY).
    poses = [fk(q) for q in path]
    tilts = np.array([np.rad2deg(np.arccos(np.clip(-P[2, 2], -1., 1.))) for P in poses])
    first = int(np.max(np.flatnonzero(np.abs(tilts - tilts[-1]) > 1e-3))) + 1
    drop = np.array([P[:3, 3] + P[:3, :3] @ offset for P in poses])[first:]
    assert len(drop) >= 50
    assert tilts[first:].max() - tilts[first:].min() < 1e-3
    assert np.max(np.abs(drop[:, :2] - drop[-1, :2])) < .001
    assert np.max(np.diff(drop[:, 2])) < 1e-6
    # A cap the wrist cannot meet is refused instead of pushed into the bound.
    with pytest.raises(ValueError, match='倾斜'):
        m.placement_attitude(base + [0., 0., .03], m.lean_azimuth(flange), joints,
                             list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values()), fk,
                             120., offset, 60., 5.)


def test_retract_above_the_frame_re_solves_the_tilt_instead_of_stopping():
    """Regression for the 2026-09-14 16:08 stop: 松爪后抬 30 mm 被 IK 拒绝.

    Holding the release attitude while rising pushes J5 past its 60° planning
    cap at this frame (5 mm -> 0.2016°, 30 mm -> 0.232286°), and the session
    asked for an emergency stop 0.45 s after 松爪. Re-solving the "as vertical
    as J5 allows" attitude at the raised height walks the same lift: 18° -> 24°
    of tilt, every commanded J5 inside the cap, XY held to the frame centre.
    """
    from grasp.geometry import pose_matrix, pose_error, transform
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    limits = list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values())
    # release_block 拿到的是 J5 限到 60° 的规划限位（执行时控制器仍按固件表）。
    plan_limits = m.cap_plan_limits(limits, 60.)
    base = np.array([0.3090757908996367, 0.08296943909941229, 0.1578650645956398])
    joints = np.array([0.3307224399604055, 1.528227746338755, -1.172983430387829,
                       -0.1460142452218456, 1.0373888008003895, 0.5026897311594068])
    flange = transform([[-0.6032581696941292, 0.07927444955406654, 0.7935963346344223,
                         0.196272],
                        [0.009518862418542412, 0.9956925508494284, -0.09222654412486508,
                         -0.019406],
                        [-0.7974891672966767, -0.04808228188073632, -0.6014142683821154,
                         0.354155],
                        [0., 0., 0., 1.]])
    offset = np.array([0., 0., .08])
    tcp45 = base + [0., 0., .045]
    rotation, q_45, tilt_45 = m.placement_attitude(tcp45, m.lean_azimuth(flange), joints,
                                                   plan_limits, fk, 120., offset, 60., 25.)
    assert tilt_45 == 18.
    start = m.flange_for_tcp(tcp45, rotation, offset)
    # 保持松爪姿态原地抬 5 mm 就已经被判据拒绝——这正是那次停机的几何。
    held = start.copy()
    held[2, 3] += .005
    with pytest.raises(ValueError, match='残差超限'):
        m.make_line_path(start, held, q_45, plan_limits, fk, 120., tcp_offset_m=offset)
    target, tilt = m.retract_pose(start, 30., q_45, plan_limits, fk, 120., offset, 60., 25.)
    assert 18. < tilt <= 25.
    path = np.asarray(m.make_line_path(start, target, q_45, plan_limits, fk, 120.,
                                       tcp_offset_m=offset), float)
    assert np.rad2deg(np.max(np.abs(path[:, 4]))) <= 60. + 1e-9
    assert np.max(np.abs(np.diff(path, axis=0))) <= np.deg2rad(.5)
    tcp = np.array([fk(q)[:3, 3] + fk(q)[:3, :3] @ offset for q in path])
    assert np.all(np.diff(tcp[:, 2]) >= -1e-9)                     # 只往上走
    assert np.max(np.abs(tcp[:, :2] - tcp[0, :2])) < .001          # XY 保持不动
    np.testing.assert_allclose(tcp[-1, 2] - tcp[0, 2], .030, atol=1e-6)
    distance, angle = pose_error(fk(path[-1]), target)
    assert distance < 1e-6 and angle < np.deg2rad(.01)
    # 倾斜上限不够时给出明确拒绝，调用方据此跳过抬离而不是请求急停。
    with pytest.raises(ValueError, match='倾斜'):
        m.retract_pose(start, 30., q_45, plan_limits, fk, 120., offset, 60., 5.)


def test_retract_plan_lowers_the_lift_when_the_tilt_budget_runs_out():
    """2026-09-14 16:33: 框在基座 X 417.6 mm，45 mm 悬停已占 23° 倾斜.

    同一帧抬 30 mm 需要 28°，默认 --max-tilt-deg 25 下整段抬离都会被拒。现在
    自动降到可行的 16 mm（倾角正好 25°）走完，不再整个跳过、也不请求急停；
    把上限放宽到 30° 则抬满 30 mm。
    """
    from grasp.geometry import pose_matrix, pose_error
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    limits = list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values())
    plan_limits = m.cap_plan_limits(limits, 60.)
    joints = np.array([0.2632480110783047, 1.8407464022008595, -1.592525675982226,
                       -0.16734216868121632, 1.0396228222429422, 0.4345695304540681])
    offset = np.array([0., 0., .08])
    start = fk(joints)
    plan, reason = m.retract_plan(start, joints, 30., offset, plan_limits, fk, 120., 60., 25.)
    assert reason == ''
    height, target, tilt, path = plan
    assert 8. <= height <= 20. and height < 30.
    assert 24. <= tilt <= 25.
    assert np.rad2deg(np.max(np.abs(path[:, 4]))) <= 60. + 1e-9
    assert np.max(np.abs(np.diff(path, axis=0))) <= np.deg2rad(.5)
    tcp = np.array([fk(q)[:3, 3] + fk(q)[:3, :3] @ offset for q in path])
    assert np.max(np.abs(tcp[:, :2] - tcp[0, :2])) < .001          # XY 只挪不到 1 mm
    assert .010 < tcp[-1, 2] - tcp[0, 2] <= height/1000 + 1e-3     # 确实往上升
    # 边界高度上 J5 正好压在 60° 规划上限，末端留下 0.1 mm 级 IK 残差（求解器
    # 本身允许 1 mm / 0.2°），执行侧的 3 mm / 1° 复核仍然通过。
    distance, angle = pose_error(fk(path[-1]), target)
    assert distance < 1e-3 and angle < np.deg2rad(.2)
    # 放宽倾斜上限就能抬满：J5 仍在 60° 规划上限内
    full, reason = m.retract_plan(start, joints, 30., offset, plan_limits, fk, 120., 60., 30.)
    assert reason == '' and full[0] == 30.
    assert 25. < full[2] <= 30.
    assert np.rad2deg(np.max(np.abs(full[3][:, 4]))) <= 60. + 1e-9
    # 上限比悬停姿态本身还紧时明确拒绝，调用方保持原位而不是急停
    refusal, reason = m.retract_plan(start, joints, 30., offset, plan_limits, fk, 120.,
                                     60., 12.)
    assert refusal is None and '倾斜' in reason


def test_one_shot_target_can_set_the_final_orientation():
    """Tool axis straight down via RPY (0, 180, yaw); the offset rotates with it."""
    from scipy.spatial.transform import Rotation
    flange = np.eye(4)
    flange[:3, :3] = Rotation.from_euler('y', 55, degrees=True).as_matrix()
    flange[:3, 3] = [.2, 0, .35]
    offset = np.array([0., 0., .08])
    target, tcp = m.tcp_target_pose(flange, offset, base_mm=[370.2, 126.5, 213.8],
                                    rpy_deg=[0., 180., 0.])
    np.testing.assert_allclose(target[:3, 2], [0., 0., -1.], atol=1e-12)
    np.testing.assert_allclose(tcp, [.3702, .1265, .2138])
    np.testing.assert_allclose(target[:3, 3], tcp - target[:3, :3] @ offset)
    with pytest.raises(ValueError, match='姿态'):
        m.tcp_target_pose(flange, offset, base_mm=[0, 0, 0], rpy_deg=[0, 0])


def test_plan_report_targets_the_clearance_above_the_frame():
    from grasp.geometry import pose_matrix
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    mdh = get_mdh('piper')
    fixed_deg = np.array(m.TARGET_DEG)
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    flange = fk(np.deg2rad(fixed_deg))
    report = m.plan_report(np.deg2rad(fixed_deg), flange, [0., 0., .25], np.eye(4),
                           [[-3.14, 3.14]]*6, fk, .030, 120., 400., [0., 0., 0.], [0., 0., .08])
    assert report['clearance_above_detected_surface_mm'] == 30.
    assert report['tcp_offset_mm'] == [0., 0., 80.]
    # The TCP target sits 100 mm above the point the camera reported.
    target = pose_matrix(report['hypothetical_target_flange_m_rad'])
    tcp = np.eye(4); tcp[:3, 3] = [0., 0., .08]
    base = np.asarray(report['red_block_base_xyz_m'])
    np.testing.assert_allclose((target @ tcp)[:3, 3], base + [0, 0, .030], atol=1e-12)
    assert report['issues'] == [] and report['IK_joints_deg'] is not None


def test_black_frame_observation_averages_several_parked_frames(monkeypatch):
    """One frame's centre scatters a few mm; the parked median is used instead."""
    points = {1.: [.10, .20, .30], 2.: [.11, .20, .30], 3.: [.10, .20, .31]}
    frames = [(stamp, stamp, stamp, None) for stamp in (1., 2., 3.)]

    class Preview:
        stage = detail = ''
        def __init__(self): self.i = 0
        def sample(self):
            value = frames[min(self.i, len(frames)-1)]
            self.i += 1
            return value
        def pump(self): pass

    monkeypatch.setattr(m, 'black_frame_point',
                        lambda prediction, frame, locate: dict(valid=True, xyz_camera_m=points[prediction]))
    monkeypatch.setattr(m.time, 'monotonic', lambda: 0.)
    controller = SimpleNamespace(tick=lambda: None, locked=False)
    sample, _, _, point = m.observe_black_frame(
        controller, fake_terminal(monkeypatch, b''), Preview(),
        lambda: (None, np.eye(4), None), lambda *a, **k: None, lambda row: None, .8)
    assert sample[0] == 3.
    np.testing.assert_allclose(point['xyz_camera_m'], [.10, .20, .30])
    assert point['observation_frames'] == 3
    assert point['observation_spread_mm'] == pytest.approx(10.)


def test_black_frame_observation_can_stay_single_frame(monkeypatch):
    frames = [(1., 1., 1., None)]
    class Preview:
        stage = detail = ''
        def sample(self): return frames[0]
        def pump(self): pass
    monkeypatch.setattr(m, 'black_frame_point',
                        lambda prediction, frame, locate: dict(valid=True, xyz_camera_m=[.1, .2, .3]))
    monkeypatch.setattr(m.time, 'monotonic', lambda: 0.)
    controller = SimpleNamespace(tick=lambda: None, locked=False)
    *_, point = m.observe_black_frame(
        controller, fake_terminal(monkeypatch, b''), Preview(),
        lambda: (None, np.eye(4), None), lambda *a, **k: None, lambda row: None, .8, frames=1)
    assert 'observation_frames' not in point and point['xyz_camera_m'] == [.1, .2, .3]


def test_missing_or_ambiguous_frame_is_reported_with_the_detector_reason():
    prediction = SimpleNamespace(boxes=[], reason='no_black_frame')
    point = m.black_frame_point(prediction, Frame(), lambda *a, **k: dict(valid=False))
    assert not point['valid'] and point['reason'] == 'black_frame_count'
    assert point['detector_reason'] == 'no_black_frame'
    # Two boxes but no intrinsics: falls back to the largest and reports as such.
    array_like = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([1, 1, 5, 5])))
    box = SimpleNamespace(xyxy=[array_like])
    point = m.black_frame_point(SimpleNamespace(boxes=[box, box], reason=None), Frame(),
                                lambda *a, **k: dict(valid=False, reason='invalid_center_depth'))
    assert point['frame_candidate']['selection'].startswith('upper first')


def test_rejected_depth_keeps_the_centre_category():
    def locate(frame, box, min_depth, max_depth):
        return dict(valid=False, reason='invalid_center_depth', pixel_uv=[10, 20])
    frame = Frame(depth=0.)
    point = m.black_frame_point(boxed(), frame, locate)
    assert point['reason'] == 'invalid_center_depth'
    assert point['center_depth_category'] == 'zero_or_negative' and point['center_raw_m'] == 0.


class Terminal:
    """Minimal fd stand-in: bytes are handed out in order, ready while any remain."""
    def __init__(self, typed):
        self.buffer = bytearray(typed)

    def select(self, rlist, wlist, xlist, timeout=0):
        return ([rlist[0]], [], []) if self.buffer else ([], [], [])

    def read(self, fd, count):
        chunk = self.buffer[:count]
        del self.buffer[:count]
        return bytes(chunk)


def fake_terminal(monkeypatch, typed):
    terminal = Terminal(typed)
    monkeypatch.setattr(m.select, 'select', terminal.select)
    monkeypatch.setattr(m.os, 'read', terminal.read)
    return terminal


@pytest.mark.parametrize('typed,exception', [
    (b'wrong\ngo\n', None), (b'quit\n', m.UserQuit),
    (b'g\x1b', KeyboardInterrupt), (b'g ', KeyboardInterrupt)])
def test_coordinate_confirmation_requires_go_or_exits(monkeypatch, typed, exception):
    controller = SimpleNamespace(tick=lambda: None, locked=False)
    terminal = fake_terminal(monkeypatch, typed)
    if exception:
        with pytest.raises(exception):
            m.wait_for_command(controller, 1, None, 'go', 'confirm')
    else:
        m.wait_for_command(controller, 1, None, 'go', 'confirm')
    assert not terminal.buffer


def test_a_mistyped_word_can_be_edited_before_enter(monkeypatch, capsys):
    """Backspace and the Delete key erase the echoed character, not just the buffer.

    Regression for the operator typing `shift` wrong: the old loop dropped the
    character from the buffer without erasing it on screen, so the word looked
    uneditable, and the Delete key (`ESC [ 3 ~`) stopped the session through its
    leading Escape.
    """
    controller = SimpleNamespace(tick=lambda: None, locked=False)
    # "shiftt" -> Backspace -> Delete -> "t" = "shift"
    fake_terminal(monkeypatch, b'shiftt\x7f\x1b[3~t\n')
    m.wait_for_command(controller, 1, None, 'shift', 'confirm')
    out = capsys.readouterr().out
    assert out.count('\b \b') == 2


def test_ctrl_u_clears_the_typed_word(monkeypatch):
    controller = SimpleNamespace(tick=lambda: None, locked=False)
    terminal = fake_terminal(monkeypatch, b'sihft\x15shift\n')
    m.wait_for_command(controller, 1, None, 'shift', 'confirm')
    assert not terminal.buffer


def test_arrow_keys_are_ignored_and_do_not_stop(monkeypatch):
    controller = SimpleNamespace(tick=lambda: None, locked=False)
    monkeypatch.setattr(m.time, 'monotonic', lambda: 1.)
    fake_terminal(monkeypatch, b'\x1b[Ago\n')
    assert m.wait_go_frame(controller, 1, 0., None, timeout_s=120.) is True


def test_go_timeout_returns_false_instead_of_stopping(monkeypatch):
    """An expired observation must re-detect, not request an emergency stop."""
    controller = SimpleNamespace(tick=lambda: None, locked=False)
    times = iter([0., 5., 500.])
    monkeypatch.setattr(m.time, 'monotonic', lambda: next(times))
    monkeypatch.setattr(m.select, 'select', lambda *a: ([], [], []))
    assert m.wait_go_frame(controller, 1, 0., None, timeout_s=120.) is False


@pytest.mark.parametrize('typed,expected,reason', [
    (b'retry\n', True, None), (b'quit\n', False, 'quit'), (b'\x04', False, 'terminal_eof')])
def test_rejection_wait_needs_retry_or_quit(monkeypatch, typed, expected, reason):
    """A rejected observation or plan re-detects or leaves; it never stops the arm."""
    events = []
    fake_terminal(monkeypatch, typed)
    preview = SimpleNamespace(pump=lambda: None)
    assert m.wait_retry_quit(None, 1, preview, lambda: None, events.append,
                             exit_event='planning_exit', retry_event='planning_retry',
                             prompt='请输入 retry 或 quit；') is expected
    expected_event = (dict(event='planning_retry') if expected
                      else dict(event='planning_exit', reason=reason))
    assert events == [expected_event]


def test_rejection_wait_keeps_the_stop_keys(monkeypatch):
    fake_terminal(monkeypatch, b'\x1b')
    preview = SimpleNamespace(pump=lambda: None)
    with pytest.raises(KeyboardInterrupt):
        m.wait_retry_quit(None, 1, preview, lambda: None, lambda row: None,
                          exit_event='planning_exit', retry_event='planning_retry',
                          prompt='请输入 retry 或 quit；')


def test_typing_go_releases_the_motion(monkeypatch):
    controller = SimpleNamespace(tick=lambda: None, locked=False)
    monkeypatch.setattr(m.time, 'monotonic', lambda: 1.)
    fake_terminal(monkeypatch, b'go\n')
    assert m.wait_go_frame(controller, 1, 0., None, timeout_s=120.) is True


def test_black_frame_point_uses_the_size_matched_candidate():
    """Two dark regions: only the one whose pixels match 70x60 mm may be used."""
    from grasp.black_frame import black_frame_point, find_black_frames
    import numpy as np
    image = np.zeros((240, 320, 3), np.uint8)
    image[:, :] = (200, 120, 40)                      # blue mat
    image[40:109, 40:120] = 20                        # 80 x 69 px  -> 70x60 mm at .35 m
    image[150:287, 60:220] = 20                       # 160 x 137 px -> too large
    frame = Frame(depth=.35)
    frame.metadata = {'intrinsics': {'fx': 400., 'fy': 400.}}
    boxes = SimpleNamespace(boxes=[], reason=None)
    class Model:
        names = {0: 'black_frame'}
    prediction = Model(); prediction.boxes = []
    from grasp.black_frame import _Box
    prediction.boxes = [_Box(c['box_xyxy']) for c in find_black_frames(image)['candidates']]
    seen = []
    def locate(frame, box, min_depth, max_depth):
        seen.append([int(v) for v in box])
        return dict(valid=True, xyz_camera_m=[0, 0, .35])
    point = black_frame_point(prediction, frame, locate)
    assert point['valid'] and seen == [[40, 40, 120, 109]]
    assert point['frame_candidate']['size_error'] < .1


def test_black_frame_point_reports_a_size_mismatch():
    from grasp.black_frame import black_frame_point
    import numpy as np
    frame = Frame(depth=.35)
    frame.metadata = {'intrinsics': {'fx': 400., 'fy': 400.}}
    prediction = SimpleNamespace(boxes=[SimpleNamespace(xyxy=[
        SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.array([0, 0, 300, 200.])))])],
        reason=None)
    point = black_frame_point(prediction, frame, lambda *a, **k: dict(valid=True))
    assert not point['valid'] and point['reason'] == 'black_frame_size_mismatch'


@pytest.mark.parametrize('argv', [['--clearance-mm', '5'],
                                  ['--clearance-mm', '200'],
                                  ['--max-travel-mm', '0'],
                                  ['--max-travel-mm', '401'],
                                  ['--max-joint-delta-deg', '0'],
                                  ['--max-joint-delta-deg', '180'],
                                  ['--pre-shift-mm', '0', '300', '0'],
                                  ['--target-base-mm', '0', '0', '0', '--target-delta-mm', '0', '0', '0'],
                                  ['--target-delta-mm', '0', '500', '0'],
                                  ['--target-base-mm', '0', '0', '2000'],
                                  ['--target-rpy-deg', '0', '180', '0'],
                                  ['--final-speed-deg-s', '0.2'],
                                  ['--follow-error-deg', '0.2'],
                                  ['--follow-error-deg', '2.5'],
                                  ['--motion-envelope-deg', '0.05'],
                                  ['--motion-envelope-deg', '2.5'],
                                  ['--path-speed-deg-s', '3', '--final-speed-deg-s', '5'],
                                  ['--target-base-mm', '370', '126', '214', '--target-rpy-deg', '0', '0', '400'],
                                  ['--hold-tolerance-deg', '0.1'],
                                  ['--target-offset-mm', '51', '0', '0'],
                                  ['--tcp-offset-mm', '0', '0', '0'],
                                  ['--tcp-offset-mm', '60', '0', '80'],
                                  ['--open-mm', '5'],
                                  ['--open-gripper', '--open-mm', '200'],
                                  ['--open-gripper', '--open-force-n', '0'],
                                  ['--open-gripper', '--open-step-mm', '0.1'],
                                  ['--open-gripper', '--open-step-ms', '20'],
                                  ['--open-gripper', '--retract-mm', '200']])
def test_entry_point_rejects_out_of_range_values(monkeypatch, argv):
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    with pytest.raises(SystemExit) as error:
        m.main(argv)
    assert error.value.code == 2
