"""Live red-block tracking -> stop/replan approach -> optional grasp and lift."""
import argparse
import json
import os
from pathlib import Path
import select
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from grasp import arm_console as ac
from grasp.hover_observation import observe_at_fixed, top_face_camera_point, wait_for_plan_retry
from grasp.hover_motion import HOLD_TOLERANCE_DEG, MOTION_TOLERANCE_DEG, timed_path, execute_path
from grasp.can_trace import CanTrace
from grasp.gripper import GripperFeedback, close_until_blocked, wait_for_gripper_feedback
from grasp.geometry import matrix_pose, pose_matrix, pose_error, transform
from grasp.hover_test import (diagnose, diagnose_best_tool_yaw, solve_target,
                              MAX_HOVER_JOINT_DELTA_DEG, TCP_OFFSET_M)
from grasp.j6_experiment import Receiver, snapshot, query_firmware
from grasp.mat_pose import TARGET_DEG, poll_key
from grasp.terminal_input import type_word
from grasp.target_tracking import (PoseHistory, TargetTracker, tracked_approach, wait_stable_target,
                                   TargetReacquisitionTimeout, TrackingReplanLimit)
from grasp.planning_process import PlanningProcess
from grasp.piper_model import piper_planning_limits
from grasp.worker_ipc import WorkerRejected
from yolo_grasp.test_hover_steps import GuardedController


# 设备身份：默认是本项目标定用的那台 D405 和那次采集会话。换相机时只设环境
# 变量即可（GRASP_CAMERA_SERIAL=xxxx、GRASP_HANDEYE_SESSION=yyyy），不用改代码，
# 但必须同步换掉 data/handeye/ 下的标定文件，否则外参对不上。
CAMERA_SERIAL = os.environ.get('GRASP_CAMERA_SERIAL', '260322275595')
LOCAL_HANDEYE_SESSION = os.environ.get('GRASP_HANDEYE_SESSION', 'd405_02')

# Grasp-and-lift leg speed.  5 deg/s keeps the lift inside the same
# following-error budget the continuous hover path already runs at.
GRASP_SPEED_DEG_S = 5.
# Continuous vision needs time to detect, stop and reacquire before contact.
TRACKING_SPEED_DEG_S = 5.
DEFAULT_TARGET_MOVE_MM = 50.
# The initial trajectory is planning round one.  Field measurements showed
# that the first replan (round two) needs this additional base-frame trim; the
# operator asked for every later round to keep the same trim instead of
# snapping back to the plain base offset.
SECOND_PLAN_EXTRA_OFFSET_MM = np.array([12., 10., 0.])
# `main` returns this when the operator asks to continue into the placement leg
# (`hold` command `place`, only offered with --continue-to-place). The combined
# entry `pick_and_place.py` turns it into "start the placement stage".
PLACE_CONTINUE = 10

# This entry deliberately uses the local hand-eye fit captured around the
# fixed grasp pose.  Its target metadata predates the later non-uniform-board
# audit; board geometry is not used at runtime, but accepting anything other
# than this exact legacy identity could silently select an unrelated result.
LOCAL_HANDEYE_BOARD = {'inner_corners': [9, 6], 'square_m': .02}


def target_offset_for_plan(base_offset_mm, plan_round):
    """Return the base-frame target trim for a one-based planning round.

    Round one uses ``base_offset_mm`` unchanged; round two and every replan
    after it add ``SECOND_PLAN_EXTRA_OFFSET_MM``.
    """
    offset = np.asarray(base_offset_mm, float)
    if offset.shape != (3,) or not np.isfinite(offset).all():
        raise ValueError('目标修正量必须是三个有限数值')
    if not isinstance(plan_round, int) or plan_round < 1:
        raise ValueError('规划轮次必须是正整数')
    return offset + (SECOND_PLAN_EXTRA_OFFSET_MM if plan_round >= 2 else 0.)


def load_local_handeye(path):
    calibration = json.loads(Path(path).read_text())
    if calibration.get('camera_serial') != CAMERA_SERIAL:
        raise ValueError('局部手眼文件的相机身份不匹配（如需换相机，设置环境变量 GRASP_CAMERA_SERIAL）')
    if calibration.get('board') != LOCAL_HANDEYE_BOARD:
        raise ValueError('局部手眼文件不是 d405_02 的9x6/20 mm候选，拒绝混用其它外参')
    # 旧标定文件里这一项是采集时的绝对路径，新文件只写会话名：两种都接受，
    # 只比较最后一段，避免把标定包死在某台机器的目录结构上。
    if Path(str(calibration.get('session', ''))).name != LOCAL_HANDEYE_SESSION:
        raise ValueError('局部手眼文件的采集会话身份不匹配')
    return calibration

def make_path(report, limits, fk, max_joint_delta_deg=MAX_HOVER_JOINT_DELTA_DEG,
              max_step_deg=.5):
    """Straight-line TCP path with an attitude hold, resampled until it fits.

    A low hover (e.g. 30 mm) needs a much larger joint excursion than the
    audited 125 mm hover, so a fixed 151-point path can exceed the per-step
    limit. The resolution grows until every step fits; the step bound itself is
    never relaxed.
    """
    if report['issues']:
        raise ValueError('坐标检查未通过：' + '；'.join(report['issues']))
    start = pose_matrix(report['current_CAN_flange_m_rad'])
    target = pose_matrix(report['hypothetical_target_flange_m_rad'])
    rotations = Rotation.from_matrix(np.stack([start[:3, :3], target[:3, :3]]))
    rotation_path = Slerp([0., 1.], rotations)
    worst = None
    for points in (151, 201, 301, 401, 601, 801, 1201):
        q = np.array(report['current_joints_rad'])
        path, worst = [], 0.
        for fraction in np.linspace(0, 1, points):
            pose = target.copy()
            pose[:3, 3] = start[:3, 3] * (1-fraction) + target[:3, 3] * fraction
            pose[:3, :3] = rotation_path([fraction]).as_matrix()[0]
            next_q = solve_target(pose, q, limits, fk, max_joint_delta_deg)
            worst = max(worst, float(np.max(abs(next_q-q))))
            if np.max(abs(next_q-np.deg2rad(TARGET_DEG))) > np.deg2rad(max_joint_delta_deg):
                raise ValueError(f'相对固定点累计关节变化超过{max_joint_delta_deg:g}°')
            path.append(next_q)
            q = next_q
        if worst <= np.deg2rad(max_step_deg):
            return path
    raise ValueError(f'路径加密到1201点仍有单步{np.rad2deg(worst):.3f}°，超过{max_step_deg:g}°')


def replan_from_current(base_point, current, current_flange, origin_flange,
                        handeye, limits, fk, args):
    """Plan from stopped feedback, retaining this attempt's original travel envelope."""
    camera_point = (np.linalg.inv(current_flange @ handeye) @ np.r_[base_point, 1])[:3]
    diagnose_fn = diagnose_best_tool_yaw if getattr(args, 'tool_yaw_search', False) else diagnose
    report = diagnose_fn(current.joints, current_flange, camera_point, handeye, limits, fk,
                         args.clearance_mm/1000, args.max_joint_delta_deg, args.max_travel_mm,
                         args.target_offset_mm/1000, args.tcp_offset_mm/1000,
                         require_fixed_start=False)
    goal = pose_matrix(report['hypothetical_target_flange_m_rad'])
    if np.linalg.norm(goal[:3, 3]-origin_flange[:3, 3]) > args.max_travel_mm/1000:
        raise ValueError('新目标超出本次抓取最初的法兰位移包络')
    path = make_path(report, limits, fk, args.max_joint_delta_deg)
    return report, timed_path(path, speed_deg_s=TRACKING_SPEED_DEG_S)


def parked_flange_tolerance(hold_tolerance_deg):
    """Flange allowance matching the parked-arm joint allowance.

    The documented 0.3-0.4 deg snap moves the flange about 1.8 mm and 0.4 deg
    (see BASE_AXES_TEST.md), so the joint allowance scales to roughly 5 mm and
    1 deg per degree of tolerance. The original 2 mm / 0.2 deg bounds remain the
    floor, so lowering --hold-tolerance-deg cannot loosen them.
    """
    return max(.002, .005*hold_tolerance_deg), np.deg2rad(max(.2, hold_tolerance_deg))


def wait_go(controller, fd, observed_at, preview=None, *, refresh_before_motion=False):
    """Wait for `go`; only the tracking flow can refresh an expired observation."""
    def poll():
        controller.tick()
        if preview is not None:
            preview.pump()
        if controller.locked:
            raise RuntimeError('监控已停止会话')
        if not refresh_before_motion and time.monotonic()-observed_at > 30:
            raise TimeoutError('观测超过30秒，拒绝使用旧目标；请重新运行')

    first = '输入 go 回车：移动到上述红块上方；空格/Esc/Ctrl+C停止。'
    while True:
        kind, typed = type_word(fd, poll, first)
        first = None
        if kind in ('stop', 'eof'):
            raise KeyboardInterrupt()
        if typed == 'go':
            return
        print('请输入 go（Backspace/Delete 可改，Ctrl+U 清空）；空格/Esc/Ctrl+C停止。', flush=True)


def describe_motion(report):
    """Print what each joint has to move, computed from the base-frame target."""
    current = np.asarray(report['current_joints_deg'], float)
    print('当前关节角(°)：' + np.array2string(current.round(3), separator=', '), flush=True)
    target = report.get('IK_joints_deg')
    if target is None:
        print('IK 未求出目标关节角：' + str(report.get('IK_error')), flush=True)
        return None
    target = np.asarray(target, float)
    delta = target-current
    flange_mm = np.asarray(report['current_CAN_flange_mm_deg'], float)[:3]
    goal_mm = np.asarray(report['hypothetical_target_flange_mm_deg'], float)[:3]
    move = goal_mm-flange_mm
    print('目标关节角(°)：' + np.array2string(target.round(3), separator=', '), flush=True)
    print('各关节需要移动(°)：' + '，'.join(f'J{index+1} {value:+.3f}'
                                        for index, value in enumerate(delta)), flush=True)
    yaw = report.get('tool_yaw_from_fixed_deg')
    orientation = ('工具轴保持朝下，绕工具轴偏航'
                   f'{yaw:+.1f}°' if yaw is not None else '姿态保持不变')
    print(f'最大单关节移动 {np.abs(delta).max():.3f}°；'
          f'法兰/TCP 基座系位移(mm)：ΔX {move[0]:+.1f}，ΔY {move[1]:+.1f}，ΔZ {move[2]:+.1f}'
          f'（{orientation}）', flush=True)
    return delta


def go_home(controller, tick, home_joints=None):
    """Return to the saved initial pose and confirm it settled.

    This controller resolves `home` to `experiment_target`, so the saved pose is
    loaded through the unbound base implementation and installed before the
    move. The step guard is re-anchored to (current pose -> initial pose) so the
    whole return stays inside the audited envelope.
    """
    if home_joints is None:
        home_joints = ac.Controller.load_home(controller)
    home_joints = np.asarray(home_joints, float)
    start = np.asarray(controller.state().joints, float)
    print('返回初始位置(°)：' + np.array2string(np.rad2deg(home_joints).round(3), separator=', '),
          flush=True)
    print('各关节需要移动(°)：' + '，'.join(
        f'J{index+1} {value:+.3f}' for index, value in enumerate(np.rad2deg(home_joints-start))),
        flush=True)
    controller.guard_start = start.copy()
    controller.guard_goal = home_joints.copy()
    controller.guard_tolerance_deg = MOTION_TOLERANCE_DEG
    controller.experiment_target = home_joints.copy()
    controller.command('home')
    while controller.active:
        tick()
        time.sleep(.02)
    final = np.asarray(controller.state().joints, float)
    error = float(np.rad2deg(np.max(abs(final-home_joints))))
    if error > .2:
        raise ValueError(f'回初始位置误差 {error:.3f}° 超过 0.2°')
    controller.guard_tolerance_deg = getattr(controller, 'hold_tolerance_deg', HOLD_TOLERANCE_DEG)
    print(f'已回到初始位置并停稳（最大关节误差 {error:.3f}°）。', flush=True)
    return home_joints


def go_to_fixed_grasp_pose(controller, tick, fixed_joints=None):
    """Return to the audited fixed grasp pose; first leg of the `home` command."""
    if fixed_joints is None:
        fixed_joints = np.deg2rad(TARGET_DEG)
    fixed_joints = np.asarray(fixed_joints, float)
    start = np.asarray(controller.state().joints, float)
    if np.max(abs(start-fixed_joints)) <= np.deg2rad(.1):
        print('已在固定抓取点，跳过第一段。', flush=True)
        return fixed_joints
    print('回固定抓取点(°)：' + np.array2string(fixed_joints.round(3), separator=', '), flush=True)
    print('各关节需要移动(°)：' + '，'.join(
        f'J{index+1} {value:+.3f}' for index, value in enumerate(np.rad2deg(fixed_joints-start))),
        flush=True)
    controller.guard_start = start.copy()
    controller.guard_goal = fixed_joints.copy()
    controller.guard_tolerance_deg = MOTION_TOLERANCE_DEG
    controller.experiment_target = fixed_joints.copy()
    controller.command('home')
    while controller.active:
        tick()
        time.sleep(.02)
    final = np.asarray(controller.state().joints, float)
    error = float(np.rad2deg(np.max(abs(final-fixed_joints))))
    if error > .2:
        raise ValueError(f'回固定抓取点误差 {error:.3f}° 超过 0.2°')
    print(f'已回固定抓取点并停稳（最大关节误差 {error:.3f}°）。', flush=True)
    return fixed_joints


def go_home_via_fixed_pose(controller, tick, home_joints=None):
    """`home` = fixed grasp pose first, then the saved initial pose.

    A single joint-space move from the hover pose straight to the initial pose
    needs up to ~90 deg per joint and really did overshoot (2026-09-11 10:46:
    J5 went 2.5 deg past its target and the hard limit stopped the session).
    Splitting the return at the audited fixed grasp pose keeps both legs small.
    """
    go_to_fixed_grasp_pose(controller, tick)
    return go_home(controller, tick, home_joints)


def handeye_audit_indices(path_length, fractions):
    """Resolve increasing, unique checkpoints on an already validated path."""
    fractions = np.asarray(fractions, float)
    if (path_length < 2 or fractions.ndim != 1 or len(fractions) < 2
            or not np.isfinite(fractions).all() or fractions[0] != 0
            or np.any(np.diff(fractions) <= 0) or fractions[-1] > .6):
        raise ValueError('手眼审计路径比例必须从0开始、严格递增且终点不超过0.6')
    indices = np.rint(fractions*(path_length-1)).astype(int)
    if np.any(np.diff(indices) <= 0):
        raise ValueError('手眼审计路径过短，无法生成不同停稳姿态')
    return indices


def run_handeye_drift_audit(controller, tracker, path, fractions, tick, event,
                            read_state, output, hold_tolerance_deg):
    """Stop at safe path fractions and measure one stationary block in base coordinates."""
    path = np.asarray(path, float)
    indices = handeye_audit_indices(len(path), fractions)
    rows = []
    for station, index in enumerate(indices):
        if station:
            segment = path[indices[station-1]:index+1]
            times, targets = timed_path(segment, speed_deg_s=TRACKING_SPEED_DEG_S)
            print(f'手眼漂移审计 {station}/{len(indices)-1}：移动到路径 {fractions[station]*100:g}% '
                  f'并停稳（预计 {times[-1]:.1f} 秒）。', flush=True)
            execute_path(controller, times, targets, tick, event,
                         hold_tolerance_deg=hold_tolerance_deg)
        tracker.preview.stage = f'Hand-eye drift audit station {station+1}/{len(indices)}'
        observation = wait_stable_target(tracker, tick, event)
        state, flange, _ = read_state()
        point = dict(observation.get('point') or observation['sample'][1].detected_point)
        base = np.asarray(observation['base'], float)
        drift = np.zeros(3) if not rows else base-np.asarray(rows[0]['base_xyz_m'])
        row = dict(station=station, path_fraction=float(fractions[station]),
                   path_index=int(index), joint_deg=np.rad2deg(state.joints).tolist(),
                   flange_m_rad=matrix_pose(flange), camera_xyz_m=point['xyz_camera_m'],
                   base_xyz_m=base.tolist(), drift_from_first_mm=(drift*1000).tolist(),
                   drift_norm_mm=float(np.linalg.norm(drift)*1000),
                   height_reference=point.get('height_reference', 'centre'),
                   top_surface_point_count=point.get('top_surface_point_count'),
                   top_surface_inlier_count=point.get('inlier_count'),
                   box_selection=point.get('selection'))
        rows.append(row)
        event(dict(event='handeye_drift_station', **row))
        print(f'  物块基座XYZ(mm)：{np.round(base*1000, 2).tolist()}；'
              f'相对首站漂移：{np.round(drift*1000, 2).tolist()} mm，'
              f'模长 {np.linalg.norm(drift)*1000:.2f} mm。', flush=True)
    report = dict(mode='stationary_block_multi_pose_handeye_drift_audit',
                  fractions=np.asarray(fractions, float).tolist(), stations=rows,
                  max_drift_mm=max(row['drift_norm_mm'] for row in rows))
    (output/'handeye_drift_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    event(dict(event='handeye_drift_audit_complete', max_drift_mm=report['max_drift_mm']))
    print(f'审计完成：最大基座坐标漂移 {report["max_drift_mm"]:.2f} mm；正在安全返回初始位置。',
          flush=True)
    go_home_via_fixed_pose(controller, tick)
    event(dict(event='home_reached', reason='handeye_drift_audit',
               joint_rad=controller.state().joints.tolist()))
    return report


def dump_trace_async(trace, path, event, registry, name):
    """Snapshot a CAN trace off the control tick.

    ``record_strike`` runs inside ``controller.tick()`` while a timed path is
    being dispatched. Serialising ~2 s of dual-channel frames there once took
    long enough to miss the 100 ms scheduling deadline and abort a healthy
    session (2026-09-18 10:17), so the copy and the write happen on a worker
    thread; ``registry`` lets the caller join them before closing the trace.
    """
    def dump():
        try:
            rows = trace.dump()
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            event(dict(event='guard_strike_dump', trace_file=name, trace_rows=len(rows)))
        except Exception as error:
            event(dict(event='can_trace_failed', error=str(error)))
    thread = threading.Thread(target=dump, name=f'guard-strike-{path.stem}', daemon=True)
    registry.append(thread)
    thread.start()
    return thread


def grasp_and_lift(controller, receiver, arm, args, tick, event, limits, fk, snapshot, *, planner=None):
    """Close the gripper on the block, then lift along base +Z.

    The gripper closes in small steps until the jaws stop following the
    command (that is "closed onto the block as far as it goes"), then the arm
    lifts ``--lift-mm`` along base +Z through the same timed-path executor and
    guards as the approach. No lift is attempted when the jaws closed with no
    block between them.
    """
    feedback = GripperFeedback(args.channel)
    try:
        feedback.start()
    except Exception as error:
        raise RuntimeError(f'夹爪反馈监听启动失败，拒绝闭合夹爪：{error}')
    try:
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        event(dict(event='gripper_close_requested', force_n=args.grip_force_n))
        first = wait_for_gripper_feedback(feedback)
        print(f'夹爪反馈已就绪：当前开度 {first["width_m"]*1000:.1f} mm。', flush=True)
        event(dict(event='gripper_feedback_ready', width_mm=first['width_m']*1000,
                   force_n=first['force_n']))
        print(f'慢慢闭合夹爪（力参数 {args.grip_force_n:g} N）：夹到物块、闭不动后停止。', flush=True)
        last_report = [0.]
        def step_event(row):
            # Keep keyboard stop and the position guard live while the jaws
            # move: closing takes ~15 s and the gripper is pressing the block.
            tick()
            if controller.locked:
                raise RuntimeError('夹爪闭合期间会话已停止')
            event(row)
            if time.time() - last_report[0] >= .5:
                last_report[0] = time.time()
                print(f'  闭合中：目标 {row["target_mm"]:.1f} mm，实际 {row["width_mm"]:.1f} mm，'
                      f'力 {row["force_n"]:.2f} N', flush=True)
        grip = close_until_blocked(
            lambda width: gripper.move_gripper_m(value=width, force=args.grip_force_n),
            lambda: feedback.latest(), args.grip_force_n,
            step_m=args.grip_step_mm/1000, step_s=args.grip_step_ms/1000, event=step_event)
        event(dict(event='gripper_closed', **grip))
        print(f'已夹住：开度 {grip["width_m"]*1000:.1f} mm，力 {grip["force_n"]:.2f} N，'
              f'{grip["steps"]} 步。', flush=True)
        _, flange0, _ = snapshot(receiver, time.time())
        lift_m = ((args.lift_mm if getattr(args, 'lift_mm', None) is not None
                   else args.lift_to_mm - args.clearance_mm) / 1000)
        target = flange0.copy()
        target[:3, 3] = target[:3, 3] + np.array([0., 0., lift_m])
        report = dict(issues=[], current_joints_rad=controller.state().joints.tolist(),
                      current_CAN_flange_m_rad=matrix_pose(flange0),
                      hypothetical_target_flange_m_rad=matrix_pose(target))
        if planner is None:  # Pure-function callers/tests without a hardware session.
            path = make_path(report, limits, fk, args.max_joint_delta_deg)
            trajectory_times, trajectory_targets = timed_path(path, speed_deg_s=GRASP_SPEED_DEG_S)
        else:
            planned = planner.run(dict(kind='path', report=report, limits=limits,
                                       max_delta=args.max_joint_delta_deg,
                                       speed_deg_s=GRASP_SPEED_DEG_S), tick)
            path = planned['path']
            trajectory_times, trajectory_targets = planned['trajectory']
        step = np.rad2deg(np.max(np.abs(np.diff(path, axis=0))))
        event(dict(event='lift_plan', lift_mm=lift_m*1000, points=len(path), max_step_deg=float(step),
                   lift_to_mm=getattr(args, 'lift_to_mm', None)))
        print(f'沿基座 +Z 抬升 {lift_m*1000:g} mm：{len(path)} 点，最大单步 {step:.3f}°，'
              f'关节目标速度≤{GRASP_SPEED_DEG_S:g}°/s，预计 {trajectory_times[-1]:.1f} 秒。', flush=True)
        # Require sustained fresh CAN updates before sending the lift trajectory.
        wait_for_fresh_feedback(receiver)
        execute_path(controller, trajectory_times, trajectory_targets, tick, event,
                     hold_tolerance_deg=args.hold_tolerance_deg)
        _, reached, _ = snapshot(receiver, time.time())
        d, a = pose_error(reached, target)
        if d > .003 or a > np.deg2rad(1):
            raise ValueError(f'抬升后法兰误差超限：{d*1000:.1f} mm / {np.rad2deg(a):.2f}°')
        after = feedback.latest()
        if abs(after['width_m'] - grip['width_m']) > .5e-3:
            raise ValueError(f'抬升期间夹爪开度从 {grip["width_m"]*1000:.1f} mm 变成 '
                             f'{after["width_m"]*1000:.1f} mm，物块可能已滑脱')
        event(dict(event='lift_reached', lift_mm=lift_m*1000, lift_to_mm=getattr(args, 'lift_to_mm', None),
                   error_mm_deg=[d*1000, float(np.rad2deg(a))],
                   width_mm=after['width_m']*1000, force_n=after['force_n'],
                   joint_rad=controller.state().joints.tolist()))
        print(f'已抬升 {lift_m*1000:g} mm（法兰误差 {d*1000:.1f} mm / {np.rad2deg(a):.2f}°；'
              f'夹爪开度 {after["width_m"]*1000:.1f} mm，力 {after["force_n"]:.2f} N）。', flush=True)
    finally:
        feedback.close()


def hold_after_motion(controller, fd, preview, event, allow_place=False, failure=None):
    """Indefinite enabled hold; display failures cannot cancel completed motion.

    With ``allow_place`` (only from the combined entry) the word ``place`` leaves
    the hold with ``PLACE_CONTINUE`` so the caller can run the placement leg: the
    arm first returns to the fixed grasp pose, then detects the black frame.
    """
    if controller.locked or controller.active is not None:
        raise RuntimeError('运动尚未正常完成，不能进入到位保持')
    event(dict(event='hold_started', joint_rad=controller.state().joints.tolist(),
               failure=None if failure is None else str(failure)))
    if failure is not None:
        print(f'本次抓取已安全停止：{failure}\n机械臂保持使能和当前位置。'
              '输入 home：先回固定抓取点、再回初始位置；quit 退出并保持位置；'
              '空格/Esc/Ctrl+C 急停。', flush=True)
    elif allow_place:
        print('已到位：保持使能和当前位置，无自动退出或保持超时。输入 place：带着物块继续放置'
              '（先回固定抓取点，再识别黑框）；home：先回固定抓取点、再回初始位置；'
              'quit 退出并保持位置；空格/Esc/Ctrl+C 急停。', flush=True)
    else:
        print('已到位：保持使能和当前位置，无自动退出或保持超时。'
              '输入 home：先回固定抓取点、再回初始位置；quit 退出并保持位置；空格/Esc/Ctrl+C 急停。', flush=True)
    preview.stage = ('Grasp stopped safely - terminal: home / quit' if failure is not None else
                     'Holding position - terminal: place / home / quit' if allow_place else
                     'Holding position - terminal: home / quit')
    preview_ok = True
    def pump():
        nonlocal preview_ok
        if not preview_ok:
            return
        try:
            preview.pump(allow_window_close=True)
        except Exception as error:
            # No camera data is needed after verified arrival. Robot feedback
            # monitoring stays outside this handler and remains fail-closed.
            preview_ok = False
            print(f'到位后预览不可用：{error}；继续保持位置并监控 CAN。', flush=True)
            event(dict(event='hold_preview_unavailable', error=str(error)))
    def tick():
        controller.tick()
        if controller.locked:
            raise RuntimeError('保持监控已停止会话，请查看 emergency_stop_requested 原因')
        pump()
    def home_tick():
        poll_key(fd)
        tick()
    while True:
        kind, typed = type_word(fd, tick)
        if kind == 'stop':
            raise KeyboardInterrupt('到位保持期间用户急停')
        if kind == 'eof':
            event(dict(event='hold_exit', reason='terminal_eof', enabled_hold=True))
            return 0
        if typed == 'quit':
            event(dict(event='hold_exit', reason='quit', enabled_hold=True))
            print('退出监控，机械臂保持使能和当前位置。', flush=True)
            return 0
        if typed == 'place':
            if not allow_place:
                print('本入口没有后续放置阶段（place 只在组合入口 pick_and_place.py 里有效）；'
                      '按 quit 处理：机械臂保持使能和当前位置。', flush=True)
                event(dict(event='hold_exit', reason='place_unavailable', enabled_hold=True))
                return 0
            event(dict(event='hold_exit', reason='place_continue', enabled_hold=True))
            print('带着物块继续放置：先回固定抓取点，到位后重新识别黑框，再等你输入 go。', flush=True)
            return PLACE_CONTINUE
        if typed == 'home':
            event(dict(event='home_requested', joint_rad=controller.state().joints.tolist()))
            go_home_via_fixed_pose(controller, home_tick)
            # Re-anchor holding guards after the explicitly requested home.
            controller.guard_start = controller.state().joints.copy()
            controller.guard_goal = controller.guard_start.copy()
            controller.guard_tolerance_deg = getattr(controller, 'hold_tolerance_deg', HOLD_TOLERANCE_DEG)
            event(dict(event='home_reached', joint_rad=controller.state().joints.tolist()))
            print('已先回固定抓取点、再回初始位置，继续保持；输入 quit 退出。', flush=True)
        else:
            words = 'place / home / quit' if allow_place else 'home / quit'
            print(f'请输入 {words}（Backspace/Delete 可改）；空格/Esc/Ctrl+C 急停。', flush=True)


def available_memory_mb():
    """MemAvailable/MemTotal in MiB, plus how much swap is already in use."""
    fields = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, _, rest = line.partition(':')
        if name in ('MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'):
            fields[name] = int(rest.strip().split()[0])//1024
    fields['SwapUsed'] = fields.get('SwapTotal', 0)-fields.get('SwapFree', 0)
    return fields


def wait_for_fresh_feedback(receiver, timeout=3., sleep=time.sleep, now=time.time):
    """Drain the CAN backlog our own heavy computation created.

    Path planning (301 IK solves) and CPU inference hold the interpreter for
    seconds, so the receive thread falls behind and every buffered frame looks
    older than the 0.25 s freshness window. Sleeping here releases the GIL and
    lets the reader catch up instead of tripping the staleness stop.
    """
    import grasp.arm_console as console
    deadline = time.monotonic()+timeout
    last = None
    fresh_since = None
    while time.monotonic() < deadline:
        try:
            # A sample just below the motion guard's 250 ms limit is still
            # backlogged. Require headroom and sustained advancing feedback.
            state = console.decode_state(receiver.snapshot(), now(), max_age=.1)
            oldest = float(state.stamps.min())
            if fresh_since is None:
                fresh_since = oldest
            if oldest - fresh_since >= .2:
                return
        except RuntimeError as error:
            last = error
            fresh_since = None
        sleep(.05)
    raise RuntimeError(f'CAN 反馈在 {timeout:g} 秒内未恢复新鲜：{last}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='can1')
    parser.add_argument('--clearance-mm', type=float, default=15,
                        help='临时 TCP 最终停在红块检测顶面上方的高度，默认15 mm（到位后夹爪才开始闭合）')
    parser.add_argument('--tracking-lock-height-mm', type=float, default=45,
                        help='TCP下降到物块顶面上方此高度后锁定当前轨迹，不再视觉重规划，默认45 mm')
    parser.add_argument('--max-joint-delta-deg', type=float, default=90,
                        help='相对固定点允许的最大单关节累计变化，默认90°（红块现场参数）')
    parser.add_argument('--max-travel-mm', type=float, default=200,
                        help='法兰总位移上限，默认200 mm（红块现场参数）')
    parser.add_argument('--hold-tolerance-deg', type=float, default=HOLD_TOLERANCE_DEG,
                        help='停在固定点等待时允许的关节偏离，默认0.8；到位后约12–16秒会自发偏移0.3–0.4°，原0.2/0.3°阈值会因此误急停。运动包络仍为0.3°，起步跟随误差仍为0.8°，调高于0.8可能被后者拒绝起步')
    parser.add_argument('--target-offset-mm', type=float, nargs=3, default=[-43., -28., 0.],
                        metavar=('DX', 'DY', 'DZ'),
                        help='基座系目标微调(mm)，默认 -43 -28 0（2026-09-18 现场确认）；Z保持0以保证TCP最终目标确为顶面上方 --clearance-mm')
    parser.add_argument('--tcp-offset-mm', type=float, nargs=3, default=None,
                        metavar=('TX', 'TY', 'TZ'),
                        help='法兰系 TCP 偏移(mm)：[横向x, 横向y, 轴向z]，默认 0 0 80；横向为 0 表示沿用“抓取点在法兰 +Z 上”的假设')
    parser.add_argument('--grasp', action='store_true',
                        help='到位后慢慢闭合夹爪直到夹住物块闭不动，再沿基座 +Z 抬到物块上方 --lift-to-mm（默认100 mm）；默认不发送任何夹爪指令')
    parser.add_argument('--grip-force-n', type=float, default=2.,
                        help='夹爪闭合力参数(N)，默认2；必须是你现场验证过的值')
    parser.add_argument('--lift-mm', type=float, default=argparse.SUPPRESS,
                        help='夹住后沿基座 +Z 抬升的相对高度(mm)；与 --lift-to-mm 二选一')
    parser.add_argument('--lift-to-mm', type=float, default=argparse.SUPPRESS,
                        help='夹住后沿基座 +Z 移动到“物块上方该高度”(mm)，默认100；必须大于 --clearance-mm')
    parser.add_argument('--grip-step-mm', type=float, default=1.,
                        help='夹爪闭合的每步开度(mm)，默认1；越小越慢越柔和')
    parser.add_argument('--grip-step-ms', type=float, default=200.,
                        help='夹爪闭合的每步节拍(ms)，默认200')
    parser.add_argument('--imgsz', type=int, choices=(320, 416, 512, 640), default=640,
                        help='YOLO 推理分辨率；显存/内存紧张时用 416 或 320')
    parser.add_argument('--device', default='0',
                        help="YOLO 推理设备：0=GPU（默认），cpu=纯CPU（完全不用 CUDA，内存紧张时用）")
    parser.add_argument('--no-window', action='store_true',
                        help='不打开相机窗口（无显示环境/省内存）；go 与停止键仍在终端')
    parser.add_argument('--target-move-mm', type=float, default=DEFAULT_TARGET_MOVE_MM,
                        help='红块基座位置连续两帧变化超过此值时停稳重规划，默认50 mm（5–50）；忽略已测得约24 mm的姿态相关局部手眼漂移，只响应大幅移动')
    parser.add_argument('--max-replans', type=int, default=5,
                        help='TCP高于视觉锁定高度时，一次接近最多自动重规划次数，默认5（1–20）')
    parser.add_argument('--min-available-mb', type=float, default=2000,
                        help='启动前要求的最小可用内存(MB)；0 表示不检查')
    parser.add_argument('--continue-to-place', action='store_true',
                        help='到位保持里额外接受 place 指令：退回调用方继续放置阶段'
                             '（组合入口 pick_and_place.py 会自动加上；单独跑本入口时没用）')
    parser.add_argument('--handeye-drift-audit', action='store_true',
                        help='不抓取：沿已验证接近路径的0/15/30/45%%停稳测量同一红块，输出手眼漂移并自动回初始位')
    args = parser.parse_args(argv)
    if args.handeye_drift_audit and (args.grasp or args.continue_to_place):
        parser.error('--handeye-drift-audit 不能与 --grasp 或 --continue-to-place 同时使用')
    if not np.isfinite(args.target_move_mm) or not 5 <= args.target_move_mm <= 50:
        parser.error('--target-move-mm 必须在5–50 mm')
    if not 1 <= args.max_replans <= 20:
        parser.error('--max-replans 必须在1–20')
    if not sys.stdin.isatty():
        parser.error('需要交互终端输入 go 和接收停止键')
    if not 10 <= args.clearance_mm <= 150:
        parser.error('高度必须为10–150 mm（原审计包络为100–150 mm）')
    if (not np.isfinite(args.tracking_lock_height_mm)
            or not args.clearance_mm < args.tracking_lock_height_mm <= 150):
        parser.error('--tracking-lock-height-mm 必须大于 --clearance-mm 且不超过150 mm')
    if not np.isfinite(args.max_travel_mm) or not 0 < args.max_travel_mm <= 200:
        parser.error('--max-travel-mm 必须在 (0,200] mm')
    if not np.isfinite(args.hold_tolerance_deg) or not .2 <= args.hold_tolerance_deg <= 10:
        parser.error('--hold-tolerance-deg 必须在 0.2–10°')
    args.target_offset_mm = np.asarray(args.target_offset_mm, float)
    if not np.isfinite(args.target_offset_mm).all() or np.max(np.abs(args.target_offset_mm)) > 50:
        parser.error('--target-offset-mm 必须为三个有限值，每个不超过 ±50 mm')
    if args.tcp_offset_mm is None:
        args.tcp_offset_mm = np.asarray(TCP_OFFSET_M, float)*1000
    args.tcp_offset_mm = np.asarray(args.tcp_offset_mm, float)
    if (not np.isfinite(args.tcp_offset_mm).all() or np.max(np.abs(args.tcp_offset_mm[:2])) > 50
            or not 0 < args.tcp_offset_mm[2] <= 200):
        parser.error('--tcp-offset-mm 横向每个不超过 ±50 mm，轴向必须在 (0,200] mm')
    args.lift_mm = getattr(args, 'lift_mm', None)
    args.lift_to_mm = getattr(args, 'lift_to_mm', None)
    if args.grasp:
        if args.grip_force_n is None or not np.isfinite(args.grip_force_n) or not 0 < args.grip_force_n <= 50:
            parser.error('--grip-force-n 必须在 (0,50] N')
        if args.lift_mm is not None and args.lift_to_mm is not None:
            parser.error('--lift-mm 与 --lift-to-mm 二选一')
        if args.lift_mm is not None:
            if not np.isfinite(args.lift_mm) or not 0 < args.lift_mm <= 100:
                parser.error('--lift-mm 必须在 (0,100] mm')
        else:
            if args.lift_to_mm is None:
                args.lift_to_mm = 100.
            if (not np.isfinite(args.lift_to_mm) or not args.clearance_mm < args.lift_to_mm <= 200):
                parser.error('--lift-to-mm 必须在 (--clearance-mm,200] mm 之间，否则不是抬升')
        if not np.isfinite(args.grip_step_mm) or not .2 <= args.grip_step_mm <= 5:
            parser.error('--grip-step-mm 必须在 0.2–5 mm')
        if not np.isfinite(args.grip_step_ms) or not 50 <= args.grip_step_ms <= 1000:
            parser.error('--grip-step-ms 必须在 50–1000 ms')
    if args.max_travel_mm > 150:
        print(f'法兰位移上限已显式扩大到 {args.max_travel_mm:g} mm，请确认整段路径空间。', flush=True)
    print(f'固定点等待偏离上限：{args.hold_tolerance_deg:g}°（运动包络仍为 {MOTION_TOLERANCE_DEG:g}°；'
          '机械臂保持约12–16秒后的0.3–0.4°自发偏移不再触发急停）。', flush=True)
    if np.any(args.target_offset_mm):
        print(f'目标微调(基座系)：ΔX {args.target_offset_mm[0]:+.1f} mm，ΔY {args.target_offset_mm[1]:+.1f} mm，'
              f'ΔZ {args.target_offset_mm[2]:+.1f} mm；这是经验修正，不是工具标定。', flush=True)
    print(f'法兰→TCP 偏移(法兰系)：x {args.tcp_offset_mm[0]:+.1f} mm，y {args.tcp_offset_mm[1]:+.1f} mm，'
          f'z {args.tcp_offset_mm[2]:+.1f} mm。', flush=True)
    if args.grasp:
        lift_mm = args.lift_mm if args.lift_mm is not None else args.lift_to_mm - args.clearance_mm
        print(f'抓取模式：到位后闭合夹爪（力 {args.grip_force_n:g} N，夹不动为止），'
              f'再沿基座 +Z 抬升 {lift_mm:g} mm'
              + (f'（到物块上方 {args.lift_to_mm:g} mm）' if args.lift_mm is None else '')
              + '；确认该力是你验证过的值、物块不会被压坏、指爪与台面无干涉。', flush=True)
    if args.hold_tolerance_deg > .8:
        print(f'注意：等待偏离上限 {args.hold_tolerance_deg:g}° 高于连续运动起步的 0.8° 跟随误差；'
              '若起步瞬间偏差仍超过 0.8°，连续运动会拒绝起步而不是自动纠正。', flush=True)
    if args.clearance_mm < 100:
        print(f'注意：{args.clearance_mm:g} mm 低于原审计包络 100–150 mm；'
              '该高度下的手指/台面间隙没有实测过，确认夹爪、相机支架和线缆不会碰到物块或台面。', flush=True)
    if args.max_joint_delta_deg > MAX_HOVER_JOINT_DELTA_DEG:
        print(f'注意：单关节累计变化上限放宽到 {args.max_joint_delta_deg:g}°（原审计值 '
              f'{MAX_HOVER_JOINT_DELTA_DEG:g}°）；这是大幅度的构型改变，确认整段路径无遮挡。', flush=True)
    memory = available_memory_mb()
    print(f'内存：可用 {memory["MemAvailable"]} MB / 共 {memory["MemTotal"]} MB；'
          f'swap 已用 {memory["SwapUsed"]} MB；YOLO 设备={args.device}，imgsz={args.imgsz}，'
          f'窗口={"关闭" if args.no_window else "开启"}。', flush=True)
    if args.device == '0':
        print('提示：GPU 推理需要 CUDA 上下文；内存紧张时改用 --device cpu（会明显变慢，但不需要 CUDA），'
              '并可用 --imgsz 320 进一步降低占用。', flush=True)
    if args.min_available_mb and memory['MemAvailable'] < args.min_available_mb:
        parser.error(
            f'可用内存 {memory["MemAvailable"]} MB 低于 {args.min_available_mb:g} MB。'
            'Jetson 是统一内存：内存不足时 CUDA/cuBLAS 初始化会失败'
            '（NvMapMemAlloc error 12 / CUBLAS_STATUS_ALLOC_FAILED）。'
            '先关闭其他 CUDA/相机程序、结束遗留的 python 进程（ps -eo pid,rss,args --sort=-rss），'
            '必要时重启后再运行；也可用 --imgsz 416 --min-available-mb 1200 降低占用')
    from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from grasp.live_depth import locate
    import cv2
    cv2.setNumThreads(1)
    from grasp.vision_process import ProcessPreview
    root = Path(__file__).resolve().parents[1]
    calibration_path = root/'data/handeye/d405_02_fixed_split/candidate_result.json'
    calibration = load_local_handeye(calibration_path)
    handeye = transform(calibration['T_flange_wrist'])
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    limits = piper_planning_limits(ROBOT_JOINT_LIMIT_PRESET_RAD)
    output = root/'yolo_grasp/runs'/('hover_go_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    print(f'悬停累计关节变化上限：{args.max_joint_delta_deg:g}°；速度3%，单步上限0.5°。', flush=True)
    print(f'本流程使用 d405_02 固定抓取姿态局部手眼外参；移动判定阈值 {args.target_move_mm:g} mm；'
          f'法兰→TCP 偏移 {np.round(args.tcp_offset_mm,1).tolist()} mm（法兰系，'
          '横向未标定、轴向为用户实测）；目标为检测表面上方。', flush=True)
    print(f'启动后自动到固定点；TCP高于物块顶面 {args.tracking_lock_height_mm:g} mm 时持续定位并可重规划；'
          f'下降到≤{args.tracking_lock_height_mm:g} mm 后锁定当前轨迹，继续到 {args.clearance_mm:g} mm。', flush=True)
    stream = dict(width=640, height=480, fps=30)
    cfg = dict(serial=CAMERA_SERIAL, color=stream, depth=stream, warmup_frames=30,
               timeout_ms=1000, global_time=True)
    preview = ProcessPreview(cfg, root/'models/red_block_best.pt', cv2, args.imgsz,
                             args.device, not args.no_window)
    planner = PlanningProcess()
    planning_parameters = dict(limits=limits, max_delta=args.max_joint_delta_deg,
                               clearance_mm=args.clearance_mm, max_travel_mm=args.max_travel_mm,
                               handeye=handeye, target_offset_mm=args.target_offset_mm,
                               tcp_offset_mm=args.tcp_offset_mm, speed_deg_s=TRACKING_SPEED_DEG_S,
                               tool_yaw_search=True)
    arm = receiver = controller = trace = None
    def create(profile):
        return AgxArmFactory.create_arm(create_agx_arm_config(robot='piper_x',
            firmeware_version=profile, interface='socketcan', channel=args.channel))
    with (output/'events.jsonl').open('x', buffering=1) as log, ac.shutdown_signals():
        log_lock = threading.Lock()
        def event(row):
            line = json.dumps(dict(host_s=time.time(), **row))+'\n'
            with log_lock:
                log.write(line)
        # Background CAN-trace writers started by the single-frame guard hook,
        # joined before the trace listener is closed.
        strike_dumps = []
        try:
            planner_info = planner.start()
            preview.start()
            while preview.sample() is None:
                preview.pump()
                time.sleep(.02)
            preview.pump()
            event(dict(event='compute_workers_started', planner_pid=planner_info['pid'],
                       vision_pid=preview.process.pid, control_pid=os.getpid()))
            arm = create('default'); arm.connect()
            firmware = query_firmware(arm); arm.disconnect()
            arm = create(resolve_firmware_profile('piper_x', firmware['software_version']))
            receiver = Receiver(args.channel); arm.connect(); time.sleep(.5)
            controller = GuardedController(arm, receiver, root/'yolo_grasp/local/arm_home.json', {}, limits, speed=3, timeout=30)
            controller.hold_tolerance_deg = args.hold_tolerance_deg
            controller.log = event
            trace = CanTrace(args.channel)
            try:
                trace.start()
            except Exception as error:
                trace = None
                print(f'注意：双通道 CAN 记录未启动（{error}）；控制流程继续，只是缺少电机帧对照。', flush=True)
                event(dict(event='can_trace_unavailable', error=str(error)))
            strike_count = 0
            def record_strike(info):
                nonlocal strike_count
                strike_count += 1
                name = f'guard_strike_{strike_count:02d}.jsonl'
                event(dict(event='guard_excursion_ignored', trace_file=name, **info))
                print(f'注意：单帧越界 {info["excursion_deg"]:+.3f}°（第{info["strike"]}帧）未停机；'
                      f'原始帧写入中 {name}。', flush=True)
                if trace is None:
                    return
                # This hook runs inside the control tick. Dumping and writing
                # ~2 s of dual-channel frames takes tens of milliseconds and
                # once pushed one tick past the 100 ms scheduling deadline,
                # aborting a healthy session. The snapshot still happens right
                # after the strike; only the disk work moves off the loop.
                dump_trace_async(trace, output/name, event, strike_dumps, name)
            controller.guard_strike_hook = record_strike
            original_stop = controller.stop
            def logged_stop(reason):
                try:
                    event(dict(event='emergency_stop_requested', reason=str(reason)))
                finally:
                    original_stop(reason)
            controller.stop = logged_stop
            original_move = arm.move_j
            def logged_move(joints):
                event(dict(event='SDK_move_j_call', joint_rad=list(joints)))
                original_move(joints)
            arm.move_j = logged_move
            with ac.keyboard() as fd:
                pose_history = PoseHistory()
                metrics_at = time.monotonic()
                last_tick = metrics_at
                max_tick_gap = 0.
                metrics_stage = preview.stage
                def tick():
                    nonlocal metrics_at, last_tick, max_tick_gap, metrics_stage
                    now = time.monotonic()
                    if metrics_stage != preview.stage:
                        metrics_stage, max_tick_gap = preview.stage, 0.
                    else:
                        max_tick_gap = max(max_tick_gap, now-last_tick)
                    last_tick = now
                    poll_key(fd); controller.tick(); preview.pump()
                    if controller.locked:
                        raise RuntimeError('会话已停止')
                    _, tracked_flange, tracked_stamps = snapshot(receiver, time.time())
                    pose_history.add(tracked_flange, tracked_stamps)
                    if now-metrics_at >= 1:
                        sample = preview.sample()
                        age = (None if sample is None else time.time()-
                               sample[1].metadata['color_timestamp_ms']/1000)
                        event(dict(event='runtime_latency', stage=preview.stage,
                                   max_tick_gap_ms=max_tick_gap*1000,
                                   vision_age_ms=None if age is None else age*1000,
                                   vision_timings_s=preview.timings, can=receiver.diagnostics()))
                        metrics_at, max_tick_gap = now, 0.
                def finish():
                    while controller.active:
                        tick(); time.sleep(.02)
                preview.stage = 'Moving to fixed pose'
                state = controller.state(); state.healthy()
                if all(state.enabled):
                    if state.ctrl_mode != 1 or state.mode_feedback != 1:
                        raise ValueError('已使能但不是CAN/J模式')
                    controller.prepared = True
                    arm._msg_mode.move_spd_rate_ctrl = 3
                elif any(state.enabled):
                    raise ValueError('六轴使能状态不一致')
                else:
                    controller.command('enable'); finish()
                controller.experiment_target = np.deg2rad(TARGET_DEG)
                controller.guard_start = controller.state().joints.copy()
                controller.guard_goal = controller.experiment_target.copy()
                controller.guard_tolerance_deg = MOTION_TOLERANCE_DEG
                print(f'以3%速度到固定点：{list(TARGET_DEG)}°', flush=True)
                controller.command('home'); finish()
                state, flange, _ = snapshot(receiver, time.time())
                if np.max(abs(state.joints-np.deg2rad(TARGET_DEG))) > np.deg2rad(.2):
                    raise ValueError('固定点到位误差超过0.2°')
                controller.guard_start = state.joints.copy(); controller.guard_goal = state.joints.copy()
                controller.guard_tolerance_deg = args.hold_tolerance_deg
                print('已到固定点，正在识别红块。', flush=True)
                planning_attempt = 0
                while True:
                    def observation_home():
                        event(dict(event='home_requested', reason='observation_rejected',
                                   joint_rad=controller.state().joints.tolist()))
                        go_home_via_fixed_pose(controller, tick)
                        event(dict(event='home_reached', reason='observation_rejected',
                                   joint_rad=controller.state().joints.tolist()))
                    observation = observe_at_fixed(
                        controller, fd, preview, lambda: snapshot(receiver, time.time()),
                        preview.names, locate, event, args.hold_tolerance_deg,
                        continuous_inference=True, home=observation_home)
                    if observation is None:
                        print('识别流程已退出，机械臂保持使能和当前位置。', flush=True)
                        return 0
                    sample, state, flange, point = observation
                    point = top_face_camera_point(point, flange @ handeye)
                    if point.get('height_reference') == 'top_face':
                        print(f'高度参考：顶面候选（{point["reference_point_count"]} 点，'
                              f'比中心像素高 {(point["reference_base_z_m"]-point["centre_reference_base_z_m"])*1000:+.1f} mm）。',
                              flush=True)
                    observed_at, frame, prediction, annotated = sample
                    planning_attempt += 1
                    attempt_output = output/f'attempt_{planning_attempt:03d}'
                    attempt_output.mkdir(exist_ok=False)
                    preview.save_sample(sample, attempt_output, tick)
                    preview.stage = 'Planning hover path'
                    preview.pump()
                    report = None
                    try:
                        planned = planner.run(dict(planning_parameters, kind='initial',
                                                   joints=state.joints, flange=flange,
                                                   camera_point=point['xyz_camera_m']), tick)
                        report, path = planned['report'], planned['path']
                        trajectory_times, trajectory_targets = planned['trajectory']
                    except WorkerRejected as error:
                        report = error.diagnostic
                        if report is not None:
                            (attempt_output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
                        event(dict(event='planning_rejected', error=str(error), diagnostic=report, attempt_dir=attempt_output.name))
                        print(f'规划拒绝：{error}', flush=True)
                        preview.detail = str(error)
                        wait_for_fresh_feedback(receiver)
                        def planning_home():
                            event(dict(event='home_requested', reason='planning_rejected',
                                       joint_rad=controller.state().joints.tolist()))
                            go_home_via_fixed_pose(controller, tick)
                            event(dict(event='home_reached', reason='planning_rejected',
                                       joint_rad=controller.state().joints.tolist()))
                        if not wait_for_plan_retry(
                                controller, fd, preview, lambda: snapshot(receiver, time.time()), state, event,
                                args.hold_tolerance_deg, continuous_inference=True,
                                home=planning_home):
                            return 0
                        continue
                    (output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
                    (attempt_output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
                    print(f"红块基座XYZ(mm)：{np.round(np.array(report['red_block_base_xyz_m'])*1000, 1)}")
                    describe_motion(report)
                    break
                steps = np.rad2deg(np.max(np.abs(np.diff(path, axis=0)), axis=1))
                print(f'路径 {len(path)} 点、最大单步 {steps.max():.3f}°、上限0.5°、速度3%；'
                      '到位后按法兰 3 mm / 1° 复核。', flush=True)
                print(f'视觉监控接近：控制轮询 50 Hz、视觉更新上限 10 Hz（实际取决于推理耗时），'
                      f'关节目标速度≤{TRACKING_SPEED_DEG_S:g}°/s，预计 {trajectory_times[-1]:.1f} 秒；'
                      f'TCP高于物块 {args.tracking_lock_height_mm:g} mm 时目标移动会中途停稳，'
                      '进入近端后不再视觉重规划。', flush=True)
                event(dict(event='plan', diagnostic=report, attempt_dir=attempt_output.name, joint_path_rad=[q.tolist() for q in path]))
                # Preserve the sustained-fresh-feedback gate before operator confirmation.
                wait_for_fresh_feedback(receiver)
                print(f'YOLO持续检测：输入go后自动监控；TCP下降到物块上方 '
                      f'{args.tracking_lock_height_mm:g} mm 前，目标移动或丢失会停稳重规划；'
                      '进入近端后按锁定轨迹完成。', flush=True)
                preview.stage = 'Waiting for go in TERMINAL - live red block detection'
                xyz = np.array(report['red_block_base_xyz_m'])*1000
                preview.detail = f'Locked base XYZ mm: {xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}'
                wait_go(controller, fd, observed_at, preview, refresh_before_motion=True)
                current, current_pose, _ = snapshot(receiver, time.time())
                d, a = pose_error(current_pose, flange)
                flange_tolerance_m, flange_tolerance_rad = parked_flange_tolerance(args.hold_tolerance_deg)
                if (np.max(abs(current.joints-state.joints)) > np.deg2rad(args.hold_tolerance_deg)
                        or d > flange_tolerance_m or a > flange_tolerance_rad):
                    raise ValueError('确认期间机械臂位置变化，目标作废')
                origin_flange = flange.copy()
                tracker = TargetTracker(preview, pose_history, handeye, preview.names, locate,
                                        threshold_m=args.target_move_mm/1000)
                if args.handeye_drift_audit:
                    print('进入手眼漂移审计：物块必须全程静止；只走路径前45%，不闭合夹爪。', flush=True)
                    try:
                        run_handeye_drift_audit(
                            controller, tracker, path, [0., .15, .30, .45], tick, event,
                            lambda: snapshot(receiver, time.time()), output,
                            args.hold_tolerance_deg)
                        print(f'审计报告：{output/"handeye_drift_audit.json"}', flush=True)
                        return 0
                    except TargetReacquisitionTimeout as error:
                        event(dict(event='handeye_drift_audit_incomplete', error=str(error),
                                   joint_rad=controller.state().joints.tolist()))
                        return hold_after_motion(controller, fd, preview, event, failure=error)
                near_path_locked = False
                replan_count = 0
                def monitor_until_lock_height():
                    nonlocal near_path_locked
                    if near_path_locked:
                        return None
                    _, live_flange, _ = snapshot(receiver, time.time())
                    live_tcp = live_flange @ np.asarray(report['provisional_T_flange_tcp'])
                    height_mm = (live_tcp[2, 3]-tracker.target[2])*1000
                    if height_mm <= args.tracking_lock_height_mm:
                        near_path_locked = True
                        preview.stage = 'Near target - current path locked, no visual replanning'
                        preview.detail = f'TCP height {height_mm:.1f} mm <= lock height'
                        event(dict(event='near_path_locked', tcp_height_above_target_mm=height_mm,
                                   lock_height_mm=args.tracking_lock_height_mm))
                        return None
                    return tracker.monitor()
                def build_replan(observation):
                    nonlocal near_path_locked, report, replan_count
                    near_path_locked = False
                    replan_count += 1
                    current, current_flange, _ = snapshot(receiver, time.time())
                    request = dict(planning_parameters, kind='replan',
                                   joints=current.joints, flange=current_flange,
                                   origin_flange=origin_flange,
                                   base_point=observation['base'])
                    plan_round = replan_count + 1
                    request['target_offset_mm'] = target_offset_for_plan(
                        args.target_offset_mm, plan_round)
                    if plan_round >= 2:
                        print(f'第{plan_round}轮规划额外基座系补偿（第二轮起沿用）：'
                              f'ΔX {SECOND_PLAN_EXTRA_OFFSET_MM[0]:+.1f} mm，'
                              f'ΔY {SECOND_PLAN_EXTRA_OFFSET_MM[1]:+.1f} mm；'
                              f'本轮总补偿 {request["target_offset_mm"].tolist()} mm。', flush=True)
                        event(dict(event='second_plan_extra_offset', plan_round=plan_round,
                                   extra_offset_mm=SECOND_PLAN_EXTRA_OFFSET_MM.tolist(),
                                   target_offset_mm=request['target_offset_mm'].tolist()))
                    result = planner.run(request, tick)
                    report = result['report']
                    return report, result['trajectory']
                try:
                    report = tracked_approach(
                        controller, tracker, report, (trajectory_times, trajectory_targets),
                        build_replan, tick, event, hold_tolerance_deg=args.hold_tolerance_deg,
                        max_replans=args.max_replans, monitor=monitor_until_lock_height,
                        verify_after_reached=False)
                except (TargetReacquisitionTimeout, TrackingReplanLimit, WorkerRejected) as error:
                    event(dict(event='grasp_recoverable_failure', error=str(error),
                               joint_rad=controller.state().joints.tolist()))
                    return hold_after_motion(controller, fd, preview, event, failure=error)
                (output/'final_tracking_diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
                _, reached, _ = snapshot(receiver, time.time())
                d, a = pose_error(reached, pose_matrix(report['hypothetical_target_flange_m_rad']))
                if d > .003 or a > np.deg2rad(1):
                    raise ValueError('上方目标最终反馈误差超限')
                tcp = reached @ np.asarray(report['provisional_T_flange_tcp'])
                gap = (tcp[:3, 3]-np.asarray(report['red_block_base_xyz_m']))*1000
                print(f'反馈计算的临时 TCP 相对物块(mm)：ΔX {gap[0]:+.1f}，ΔY {gap[1]:+.1f}，ΔZ {gap[2]:+.1f}；不是实测指尖间隙。', flush=True)
                event(dict(event='hover_reached', provisional_tcp_offset_from_block_mm=gap.tolist()))
                if args.grasp:
                    grasp_and_lift(controller, receiver, arm, args, tick, event, limits, fk, snapshot,
                                   planner=planner)
                status = hold_after_motion(controller, fd, preview, event,
                                           allow_place=args.continue_to_place)
                if status == PLACE_CONTINUE:
                    return PLACE_CONTINUE
                return status
        except BaseException as error:
            event(dict(event='aborted', error=str(error)))
            if trace is not None:
                try:
                    rows = trace.dump()
                    (output/'can_trace_abort.jsonl').write_text(
                        ''.join(json.dumps(row)+'\n' for row in rows))
                    event(dict(event='can_trace_abort', rows=len(rows), file='can_trace_abort.jsonl'))
                except Exception as dump_error:
                    event(dict(event='can_trace_failed', error=str(dump_error)))
            if controller is not None and not controller.locked:
                controller.stop(str(error))
            print(f'流程停止：{error}', flush=True)
            return 2
        finally:
            for thread in strike_dumps:
                thread.join(timeout=2)
            if trace is not None:
                trace.close()
            if receiver is not None: receiver.close()
            if arm is not None: arm.disconnect()
            try:
                preview.close()
            finally:
                planner.close()


if __name__ == '__main__':
    raise SystemExit(main())
