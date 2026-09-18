"""Fixed pose -> detect the black frame on the blue mat -> hover above it.

Standalone placement leg: it never touches the gripper and does not import the
grasp entry's behaviour beyond the shared move/guard helpers. Enter `go` only
after the target is locked; `quit` exits, `home` returns through the fixed
grasp pose. The arm keeps holding at the end.
"""
import argparse
import json
import os
from pathlib import Path
import select
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from grasp import arm_console as ac
from grasp.black_frame import BlackFrameModel, black_frame_point
from grasp.can_trace import CanTrace
from grasp.geometry import matrix_pose, pose_error, pose_matrix, transform
from grasp.hover_motion import HOLD_TOLERANCE_DEG, MOTION_TOLERANCE_DEG, timed_path, execute_path
from grasp.hover_test import MAX_HOVER_JOINT_DELTA_DEG, TCP_OFFSET_M, hover_target, solve_target
from grasp.j6_experiment import Receiver, snapshot, query_firmware
from grasp.gripper import GripperFeedback, wait_for_gripper_feedback
from grasp.mat_pose import TARGET_DEG, configured_max_width, poll_key
from grasp.piper_model import piper_planning_limits
from yolo_grasp.test_hover_steps import GuardedController
from yolo_grasp import hover_red_block as hrb
from yolo_grasp.hover_red_block import (available_memory_mb, go_home_via_fixed_pose, make_path,
                                        parked_flange_tolerance, wait_for_fresh_feedback)


class UserQuit(Exception):
    """Operator asked to leave before any motion was authorised."""


# Cap for the live black-frame detection while the pre-planned path runs: enough
# for the operator to watch the frame, small enough that the CAN reader thread and
# the control tick keep their scheduling time (see HoverPreview.min_interval).
LIVE_DETECTION_INTERVAL_S = .25
# Frames averaged for the placement target while the arm is parked at the fixed
# pose; see observe_black_frame for the run-to-run scatter this removes.
PLACE_OBSERVATION_FRAMES = 3
PLACE_OBSERVATION_TIMEOUT_S = 3.
# Firmware speed percentage for the placement leg's streamed paths. It must be
# at least as fast as the planned °/s (3 % was ~5.2 °/s, i.e. exactly the old
# 5 °/s plan); 5 % leaves headroom for the faster 7/5 °/s defaults so the arm
# follows instead of lagging into the following-error guard.
PLACE_SPEED_PERCENT = 5


def flange_for_tcp(tcp, rotation, tcp_offset_m):
    """Flange pose that puts the TCP at ``tcp`` with a given flange rotation."""
    pose = np.eye(4)
    pose[:3, :3] = np.asarray(rotation, float)
    pose[:3, 3] = np.asarray(tcp, float) - pose[:3, :3] @ np.asarray(tcp_offset_m, float)
    return pose


def clip_seed(joints, limits):
    """Clip a seed onto the joint bounds.

    With J5 capped at its real travel the IK lands *on* the bound, and the value
    it returns can sit a nanodegree outside it; feeding that back as the next
    seed would make scipy reject ``x0`` as infeasible and abort the path.
    """
    bounds = np.asarray(limits, float)
    return np.clip(np.asarray(joints, float), bounds[:, 0], bounds[:, 1])


def cap_plan_limits(limits, max_j5_deg):
    """Copy of the joint table with J5 capped for *planning* only.

    The cap must not reach the controller: its ``validate_joints`` compares the
    raw feedback against the table with no tolerance, so a 0.3-0.4° joint-frame
    glitch around the cap (documented for this arm) would be read as 关节越界 and
    stop the session -- on 2026-09-14 that turned a `home` request into a
    misleading "回固定抓取点误差 50°" abort. Planning uses the capped copy, the
    controller keeps the firmware limits.
    """
    capped = [list(row) for row in np.asarray(limits, float)]
    capped[4] = [-np.deg2rad(abs(max_j5_deg)), np.deg2rad(abs(max_j5_deg))]
    return capped


def open_steps(current_m, target_m, step_m):
    """Width targets from ``current_m`` up to ``target_m`` in ``step_m`` steps.

    Opening cannot stall on a block the way closing does, so the sequence is
    planned up front: the jaws are commanded a little wider each step and the
    feedback only has to keep up. Returns at least one target.
    """
    if not (0 <= current_m <= 1) or not (0 < target_m <= 1) or not (0 < step_m <= .05):
        raise ValueError('张爪宽度或步长无效')
    if target_m <= current_m:
        return [target_m]
    steps, width = [], current_m
    while width < target_m - 1e-9:
        width = min(target_m, width + step_m)
        steps.append(width)
    return steps


def release_block(controller, arm, receiver, args, tick, event, limits, fk, snapshot,
                  tcp_offset_m):
    """Open the gripper at the hover point, then lift clear of the frame.

    Reuses the grasp leg's gripper feedback (the same 0x2A8 listener and the same
    "keep ticking while the jaws move" loop) but steps the width *up*, checks that
    it really opened, and then retracts along base +Z through the shared timed path
    and guards so the open jaws do not drag through the frame on the way to `home`.
    Only runs with `--open-gripper`.
    """
    if not np.isfinite(args.open_force_n) or not 0 < args.open_force_n <= 50:
        raise ValueError('张爪力参数必须在 (0,50] N')
    feedback = GripperFeedback(args.channel)
    try:
        feedback.start()
    except Exception as error:
        raise RuntimeError(f'夹爪反馈监听启动失败，拒绝张爪：{error}')
    try:
        gripper = arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
        first = wait_for_gripper_feedback(feedback)
        maximum = configured_max_width(gripper)
        target_m = maximum if args.open_mm is None else args.open_mm/1000
        if target_m > maximum + 1e-6:
            raise ValueError(f'张爪目标 {target_m*1000:.1f} mm 超过设备最大行程 {maximum*1000:g} mm')
        print(f'到位开度 {first["width_m"]*1000:.1f} mm，目标 {target_m*1000:.1f} mm，'
              f'力参数 {args.open_force_n:g} N，每步 {args.open_step_mm:g} mm。', flush=True)
        event(dict(event='gripper_open_requested', target_mm=target_m*1000,
                   force_n=args.open_force_n, width_mm=first['width_m']*1000))
        state = first
        for index, width_m in enumerate(open_steps(first['width_m'], target_m,
                                                   args.open_step_mm/1000), 1):
            tick()
            if controller.locked:
                raise RuntimeError('张爪期间会话已停止')
            gripper.move_gripper_m(value=width_m, force=args.open_force_n)
            step_end = time.monotonic() + args.open_step_ms/1000
            while time.monotonic() < step_end:
                tick(); time.sleep(.02)
            state = feedback.latest()
            if state is None:
                raise RuntimeError('张爪期间缺少夹爪反馈')
            event(dict(event='gripper_open_step', target_mm=width_m*1000,
                       width_mm=state['width_m']*1000, force_n=state['force_n'], step=index))
            print(f'  张开中：目标 {width_m*1000:.1f} mm，实际 {state["width_m"]*1000:.1f} mm，'
                  f'力 {state["force_n"]:.2f} N', flush=True)
        if state['width_m'] < first['width_m'] + .005:
            raise RuntimeError(f'张爪未确认：开度 {first["width_m"]*1000:.1f} -> '
                               f'{state["width_m"]*1000:.1f} mm')
        event(dict(event='gripper_opened', width_mm=state['width_m']*1000, force_n=state['force_n']))
        print(f'已松开：开度 {state["width_m"]*1000:.1f} mm。', flush=True)
    finally:
        feedback.close()
    if args.retract_mm <= 0:
        return
    _, flange, _ = snapshot(receiver, time.time())
    start = np.array(flange, float, copy=True)
    joints_now = controller.state().joints.copy()
    plan, reason = retract_plan(start, joints_now, args.retract_mm, tcp_offset_m, limits, fk,
                                args.max_joint_delta_deg, args.max_j5_deg, args.max_tilt_deg)
    if plan is None:
        # A rejected lift must not request an emergency stop: the arm can only
        # stay where it released the block, which is exactly the state of
        # --retract-mm 0, and the operator decides what happens next.
        event(dict(event='retract_skipped', retract_mm=args.retract_mm, error=reason))
        print(f'松爪后抬不起来（{reason}）；机械臂保持原位、继续监控。'
              '可放宽 --max-tilt-deg 或减小 --clearance-mm 后重试。', flush=True)
        # The search blocks without ticking, so the receive thread falls behind.
        # Returning straight into the hold loop then trips the 0.25 s freshness
        # check on its very first tick (2026-09-14 16:33: "CAN 反馈过期或时钟
        # 不一致" arrived 1 ms after the event). Drain the backlog like the
        # successful lift path does before handing control back.
        wait_for_fresh_feedback(receiver)
        return
    height_mm, target, tilt, path = plan
    if height_mm < args.retract_mm - 1e-9:
        event(dict(event='retract_reduced', requested_mm=args.retract_mm, used_mm=height_mm,
                   tool_tilt_deg=float(tilt)))
        print(f'松爪后抬不到 {args.retract_mm:g} mm（该高度需要更大的工具轴倾斜），'
              f'已自动降到可行的 {height_mm:.1f} mm。', flush=True)
    times, targets = timed_path(path, args.path_speed_deg_s)
    step = np.rad2deg(np.max(np.abs(np.diff(path, axis=0))))
    event(dict(event='retract_plan', retract_mm=args.retract_mm, used_mm=height_mm,
               points=len(path), max_step_deg=float(step), duration_s=float(times[-1]),
               tool_tilt_deg=float(tilt)))
    print(f'松爪后沿基座 +Z 抬离 {height_mm:g} mm（工具轴倾到 {tilt:.0f}°）：{len(path)} 点，'
          f'最大单步 {step:.3f}°，{args.path_speed_deg_s:g}°/s，预计 {times[-1]:.1f} 秒。', flush=True)
    wait_for_fresh_feedback(receiver)
    execute_path(controller, times, targets, tick, event,
                 hold_tolerance_deg=args.hold_tolerance_deg,
                 follow_error_deg=args.follow_error_deg,
                 motion_tolerance_deg=args.motion_envelope_deg,
                 speed_percent=PLACE_SPEED_PERCENT)
    _, reached, _ = snapshot(receiver, time.time())
    d, a = pose_error(reached, target)
    if d > .003 or a > np.deg2rad(1):
        raise ValueError(f'抬离后法兰误差超限：{d*1000:.1f} mm / {np.rad2deg(a):.2f}°')
    event(dict(event='retract_reached', error_mm_deg=[d*1000, float(np.rad2deg(a))],
               joint_rad=controller.state().joints.tolist()))
    print(f'已抬离（法兰误差 {d*1000:.1f} mm / {np.rad2deg(a):.2f}°）。', flush=True)


def placement_attitude(tcp, azimuth, seed, limits, fk, max_delta_deg, tcp_offset_m,
                       max_j5_deg=60., max_tilt_deg=25., step_deg=1., min_tilt_deg=0.):
    """Placement attitude: as close to straight down as J5 allows.

    This arm's J5 jams a little under 70° of its reported travel, and holding the
    tool axis straight down at this frame needs about 87° of it (2026-09-14: three
    runs stopped there with the servo pushing 12 A into the bound). The tool axis
    is therefore tilted *away* in ``azimuth`` (the direction the current attitude
    already leans) until the IK fits inside ±``max_j5_deg``, and the tilt is kept
    at or below ``max_tilt_deg``. Returns ``(rotation, joints, tilt_deg)``.
    """
    capped = np.asarray(limits, float).copy()
    capped[4] = [-np.deg2rad(abs(max_j5_deg)), np.deg2rad(abs(max_j5_deg))]
    down = np.diag([-1., 1., -1.])
    # ``min_tilt_deg`` lets the retract search skip tilts the arm already holds
    # (each costs one IK solve); the placement itself starts at 0. The grid stays
    # anchored at 0 so no candidate can overshoot ``max_tilt_deg``.
    for tilt_deg in np.arange(0., max_tilt_deg + step_deg/2, step_deg):
        if tilt_deg < min_tilt_deg - 1e-9:
            continue
        rotation = _rot_z(azimuth) @ _rot_y(-np.deg2rad(tilt_deg)) @ down
        pose = flange_for_tcp(tcp, rotation, tcp_offset_m)
        try:
            joints = solve_target(pose, clip_seed(seed, capped), capped, fk, max_delta_deg)
        except ValueError:
            continue
        return rotation, clip_seed(joints, limits), float(tilt_deg)
    raise ValueError(f'J5 限制在 {max_j5_deg:g}° 以内时，工具轴需要倾斜超过 {max_tilt_deg:g}° 才能到位；'
                     '可放宽 --max-j5-deg 或 --max-tilt-deg，或改换落点')


def retract_pose(flange, retract_mm, seed, limits, fk, max_delta_deg, tcp_offset_m,
                 max_j5_deg=60., max_tilt_deg=25., min_tilt_deg=0.):
    """Flange pose ``retract_mm`` above the current TCP with the least tilt J5 allows.

    Keeping the release attitude while rising costs more J5 the higher the TCP
    goes, and at the placement point the wrist already sits at the planning cap:
    on 2026-09-14 16:08 the fixed-attitude lift was rejected 0.45 s after 松爪
    (抬 5 mm 姿态0.2016°、抬 30 mm 姿态0.2312°) and requested an emergency stop.
    Re-solving the placement attitude at the raised height -- the same "as
    vertical as J5 allows" rule the descent uses -- keeps the joint inside the
    cap: the same frame asks for 18° at 45 mm and 24° at 75 mm, 151 points with
    a 0.026° worst step and J5 ≤ 59.7°. Returns ``(flange_pose, tilt_deg)``.
    """
    start = np.asarray(flange, float)
    offset = np.asarray(tcp_offset_m, float)
    tcp_goal = start[:3, 3] + start[:3, :3] @ offset + np.array([0., 0., retract_mm/1000])
    rotation, _, tilt = placement_attitude(tcp_goal, lean_azimuth(start), seed, limits, fk,
                                           max_delta_deg, offset, max_j5_deg, max_tilt_deg,
                                           min_tilt_deg=min_tilt_deg)
    return flange_for_tcp(tcp_goal, rotation, offset), tilt


def retract_plan(start, seed, retract_mm, tcp_offset_m, limits, fk, max_delta_deg,
                 max_j5_deg=60., max_tilt_deg=25., min_mm=1.):
    """Longest feasible ``base +Z`` lift up to ``retract_mm`` and its joint path.

    The wrist needs more J5 the higher the TCP goes while the XY stays put, so a
    lift that is out of reach usually still has a shorter feasible version: on
    2026-09-14 16:33 the frame sat at 417.6 mm (X) where the 45 mm hover already
    used 23° of tilt, so the requested 30 mm lift needed 28° and was refused by
    the 25° cap. The search bisects on the height instead of stepping one
    millimetre at a time, and skips tilts the arm already holds, so a refused
    lift costs a few tenths of a second rather than one IK search per millimetre
    (each of those blocks the CAN receiver).

    Returns ``(height_mm, flange_pose, tilt_deg, path)``, or ``(None, reason)``
    when even ``min_mm`` cannot be planned.
    """
    start = np.asarray(start, float)
    tilt_now = float(np.rad2deg(np.arccos(np.clip(-start[2, 2], -1., 1.))))
    floor = min(tilt_now, float(max_tilt_deg))
    last_error = ['']

    def probe(height_mm):
        if height_mm < min_mm:
            return None
        try:
            target, tilt = retract_pose(start, height_mm, seed, limits, fk, max_delta_deg,
                                        tcp_offset_m, max_j5_deg, max_tilt_deg, floor)
            path = np.asarray(make_line_path(start, target, seed, limits, fk, max_delta_deg,
                                             tcp_offset_m=tcp_offset_m), float)
        except ValueError as error:
            last_error[0] = str(error)
            return None
        return float(height_mm), target, float(tilt), path

    best = probe(retract_mm)
    if best is not None:
        return best, ''
    if retract_mm <= min_mm:
        return None, (last_error[0] or '抬离不可行')
    low, high = float(min_mm), float(retract_mm)      # low unproven, high refused
    while high - low > 1.:
        middle = .5*(low + high)
        candidate = probe(middle)
        if candidate is None:
            high = middle
        else:
            low, best = middle, candidate
    if best is None:
        best = probe(low)
    if best is not None:
        return best, ''
    return None, (last_error[0] or '抬离不可行')


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def lean_azimuth(flange):
    """Direction the current tool axis leans in the base frame (radians)."""
    axis = np.asarray(flange, float)[:3, 2]
    return float(np.arctan2(axis[1], axis[0]))


def tcp_target_pose(flange, tcp_offset_m, *, base_mm=None, delta_mm=None, rpy_deg=None):
    """Flange target (orientation kept) that puts the TCP at a base-frame point.

    ``base_mm`` is an absolute TCP coordinate, ``delta_mm`` is relative to the
    current TCP. Exactly one of them must be given. ``rpy_deg`` optionally sets
    the final TCP orientation in the base frame (same xyz-Euler convention as
    ``grasp.geometry.pose_matrix``); without it the current orientation is kept.
    """
    flange = np.asarray(flange, float)
    offset = np.asarray(tcp_offset_m, float)
    tcp_now = flange[:3, 3] + flange[:3, :3] @ offset
    if (base_mm is None) == (delta_mm is None):
        raise ValueError('必须且只能给出 --target-base-mm 或 --target-delta-mm')
    wanted = np.asarray(base_mm if base_mm is not None else delta_mm, float)
    if wanted.shape != (3,) or not np.isfinite(wanted).all():
        raise ValueError('目标必须为三个有限毫米值')
    tcp = wanted/1000 if base_mm is not None else tcp_now + wanted/1000
    target = np.array(flange, copy=True)
    if rpy_deg is not None:
        rpy = np.asarray(rpy_deg, float)
        if rpy.shape != (3,) or not np.isfinite(rpy).all():
            raise ValueError('目标姿态必须为三个有限角度(deg)')
        from scipy.spatial.transform import Rotation
        target[:3, :3] = Rotation.from_euler('xyz', np.deg2rad(rpy)).as_matrix()
    target[:3, 3] = tcp - target[:3, :3] @ offset
    return target, tcp


def make_line_path(start_pose, target_pose, start_joints, limits, fk, max_joint_delta_deg,
                   max_step_deg=.5, tcp_offset_m=(0., 0., 0.)):
    """Straight TCP line with the orientation slerped from start to target.

    The shared ``make_path`` holds the target orientation for every waypoint,
    which is fine while the orientation does not change but would demand one
    huge first step when it does (e.g. --target-rpy-deg 0 180 0). The resolution
    grows until every step fits; the step bound itself is never relaxed.
    """
    from scipy.spatial.transform import Rotation, Slerp
    start_pose = np.asarray(start_pose, float)
    target_pose = np.asarray(target_pose, float)
    start_joints = np.asarray(start_joints, float)
    r0 = Rotation.from_matrix(start_pose[:3, :3])
    r1 = Rotation.from_matrix(target_pose[:3, :3])
    interp = Slerp([0., 1.], Rotation.concatenate([r0, r1]))
    offset = np.asarray(tcp_offset_m, float)
    start_tcp = start_pose[:3, 3] + start_pose[:3, :3] @ offset
    target_tcp = target_pose[:3, 3] + target_pose[:3, :3] @ offset
    worst = 0.
    for points in (151, 201, 301, 401, 601, 801, 1201):
        q = clip_seed(start_joints, limits)
        path, worst = [], 0.
        for fraction in np.linspace(0, 1, points):
            pose = np.eye(4)
            pose[:3, :3] = interp(fraction).as_matrix()
            pose[:3, 3] = start_tcp*(1-fraction) + target_tcp*fraction - pose[:3, :3] @ offset
            next_q = solve_target(pose, clip_seed(q, limits), limits, fk, max_joint_delta_deg)
            worst = max(worst, float(np.max(abs(next_q-q))))
            if np.max(abs(next_q-start_joints)) > np.deg2rad(max_joint_delta_deg):
                raise ValueError(f'相对起点累计关节变化超过{max_joint_delta_deg:g}°')
            path.append(next_q)
            q = next_q
        if worst <= np.deg2rad(max_step_deg):
            return path
    raise ValueError(f'路径加密到1201点仍有单步{np.rad2deg(worst):.3f}°，超过{max_step_deg:g}°')


def tool_down_reach(target, end_tcp, max_height_m, joints, limits, fk, max_delta, offset,
                    step_m=.001):
    """Highest point above the placement target that still takes a tool-down pose.

    Holding the tool axis straight down forces the wrist to bend more the higher
    the TCP is, and this arm runs into J5 (±89°) barely above the placement point:
    with the frame ~0.40 m from the base the tool-down attitude stops being
    solvable ~20 mm above the final TCP. The 2026-09-14 plan asked for it 113 mm
    above the target, so the IK came back with a 0.31° residual (J5 clamped at
    88.9°) and the session aborted. The reach depends on where the frame is
    detected, so it is searched per plan instead of hard-coded, and heights below
    the result are known solvable because they were solved on the way up.
    """
    base = np.asarray(end_tcp, float)
    seed = np.asarray(joints, float)
    reach = None
    for height in np.arange(0., float(max_height_m) + step_m/2, step_m):
        pose = np.array(target, copy=True)
        pose[:3, 3] = np.r_[base[:2], base[2] + height] - pose[:3, :3] @ offset
        try:
            seed = solve_target(pose, clip_seed(seed, limits), limits, fk, max_delta)
        except ValueError:
            break
        reach = height
    if reach is None:
        raise ValueError('放置目标工具轴竖直的姿态不可解（IK 残差或关节变化超限）')
    return reach


def vertical_paths(start, target, joints, limits, fk, max_delta, offset, max_travel_mm,
                   overhead_m=.070, max_tilt_deg=25.):
    """Plan a level approach plus a fixed-XY descent that ends with the tool low.

    The tool axis cannot be turned down above the target (see
    ``tool_down_reach``), so the turn is shared with the descent instead of being
    completed on the way: the level approach keeps the current attitude, the
    descent turns the tool axis down at the highest height the wrist still takes
    it, and the remaining straight drop already has the target attitude. The tool
    axis may stay tilted by up to ``max_tilt_deg`` when J5 cannot hold it straight
    down; the final pose, the clearance above the detected frame and the TCP route
    (level, then straight down) are unchanged, only where the attitude turns -- and
    therefore how long the pure vertical drop is -- follows the joint limits.
    """
    offset = np.asarray(offset, float)
    start, target = np.asarray(start, float), np.asarray(target, float)
    tilt_deg = float(np.rad2deg(np.arccos(np.clip(-target[2, 2], -1., 1.))))
    if not np.isfinite(tilt_deg) or tilt_deg > max_tilt_deg + 1e-9:
        raise ValueError(f'放置目标工具轴必须朝下且与竖直夹角不超过 {max_tilt_deg:g}°'
                         f'（当前 {tilt_deg:.1f}°）')
    start_tcp = start[:3, 3] + start[:3, :3] @ offset
    end_tcp = target[:3, 3] + target[:3, :3] @ offset
    overhead_tcp = end_tcp.copy()
    overhead_tcp[2] = max(start_tcp[2], end_tcp[2] + overhead_m)
    overhead = np.array(target, copy=True)
    overhead[:3, :3] = start[:3, :3]
    overhead[:3, 3] = overhead_tcp - overhead[:3, :3] @ offset
    approach = make_line_path(start, overhead, joints, limits, fk, max_delta,
                              tcp_offset_m=offset)
    reach = tool_down_reach(target, end_tcp, overhead_tcp[2] - end_tcp[2], joints, limits,
                            fk, max_delta, offset)
    turn = np.array(target, copy=True)
    turn[:3, 3] = np.r_[end_tcp[:2], end_tcp[2] + reach] - target[:3, :3] @ offset
    descent = make_line_path(overhead, turn, approach[-1], limits, fk, max_delta,
                             tcp_offset_m=offset)
    drop = make_line_path(turn, target, descent[-1], limits, fk, max_delta,
                          tcp_offset_m=offset)
    for q in approach + descent[1:] + drop[1:]:
        if np.max(abs(q - joints)) > np.deg2rad(max_delta):
            raise ValueError('放置路径累计关节变化超限')
        if np.linalg.norm(fk(q)[:3, 3] - start[:3, 3]) > max_travel_mm/1000:
            raise ValueError('放置路径法兰位移超过上限')
    return approach, descent + drop[1:]


def move_to_pose(controller, tick, read_state, target, limits, fk, max_joint_delta_deg,
                 event, label):
    """One joint-space SDK move (3%) to ``target`` with guard and pose check."""
    state, flange, _ = read_state()
    q_target = solve_target(target, state.joints, limits, fk, max_joint_delta_deg)
    delta = np.rad2deg(q_target-state.joints)
    event(dict(event=f'{label}_plan', target_flange_m_rad=matrix_pose(target),
               joint_rad=state.joints.tolist(), target_joint_rad=q_target.tolist()))
    print(f'{label} 各关节需要移动(°)：' + '，'.join(
        f'J{i+1} {value:+.1f}' for i, value in enumerate(delta)), flush=True)
    controller.guard_start = state.joints.copy()
    controller.guard_goal = q_target.copy()
    controller.guard_tolerance_deg = MOTION_TOLERANCE_DEG
    controller.experiment_target = q_target.copy()
    started = time.monotonic()
    controller.command('home')
    while controller.active:
        tick(); time.sleep(.02)
    _, reached, _ = read_state()
    distance, angle = pose_error(reached, target)
    if distance > .003 or angle > np.deg2rad(1):
        raise ValueError(f'{label} 后法兰误差超限：{distance*1000:.1f} mm / {np.rad2deg(angle):.2f}°')
    event(dict(event=f'{label}_reached', error_mm_deg=[distance*1000, float(np.rad2deg(angle))],
               joint_rad=controller.state().joints.tolist()))
    print(f'{label} 完成（用时 {time.monotonic()-started:.1f} 秒，'
          f'法兰误差 {distance*1000:.1f} mm / {np.rad2deg(angle):.2f}°）。', flush=True)
    return reached


def shifted_flange(flange, shift_mm):
    """Flange pose after moving the TCP by a base-frame shift (orientation kept).

    The TCP offset is rigid, so shifting the TCP by a base-frame vector is the
    same as shifting the flange by that vector while holding the orientation.
    """
    shift = np.asarray(shift_mm, float)
    if shift.shape != (3,) or not np.isfinite(shift).all() or np.max(np.abs(shift)) > 200:
        raise ValueError('预移动量必须为三个有限毫米值，每个不超过 ±200 mm')
    target = np.array(flange, float, copy=True)
    target[:3, 3] = target[:3, 3] + shift/1000
    return target



# Terminal word input lives in one shared module so the grasp leg's prompts
# (retry / go) accept Backspace/Delete exactly like the placement leg's.
from grasp.terminal_input import type_word, word_key              # noqa: E402


def wait_for_command(controller, fd, preview, word, prompt):
    """Wait for one typed word; stop keys and `quit` stay available."""
    def poll():
        controller.tick()
        if preview is not None:
            preview.pump()
        if controller.locked:
            raise RuntimeError('会话已停止')

    while True:
        kind, typed = type_word(fd, poll, prompt)
        if kind in ('stop', 'eof'):
            raise KeyboardInterrupt()
        if typed == word:
            return
        if typed == 'quit':
            raise UserQuit()
        print(f'请输入 {word} 或 quit（Backspace/Delete 可改，Ctrl+U 清空）；尚未发送运动目标。',
              flush=True)


def wait_go_frame(controller, fd, observed_at, preview=None, timeout_s=120.):
    """Return True on `go`, False when the observation expired (re-detect)."""
    def poll():
        controller.tick()
        if preview is not None:
            preview.pump()
        if controller.locked:
            raise RuntimeError('会话已停止')

    # Expiry returns to the detection loop instead of stopping the session: a
    # timeout is not a fault and must not request an emergency stop. It is
    # measured from the observation, so planning time counts against the window.
    deadline = observed_at + timeout_s
    while True:
        kind, typed = type_word(
            fd, poll, '输入 go 回车：移动到上述黑色框上方；空格/Esc/Ctrl+C停止。', deadline)
        if kind == 'timeout':
            print(f'观测超过 {timeout_s:g} 秒未确认：目标作废，重新识别（未发送任何运动目标）。', flush=True)
            return False
        if kind in ('stop', 'eof'):
            raise KeyboardInterrupt()
        if typed == 'go':
            return True
        print('请输入 go（Backspace/Delete 可改）；空格/Esc/Ctrl+C 停止，尚未发送运动目标。', flush=True)


def plan_report(joints, flange, camera_point, handeye, limits, fk, clearance, max_delta_deg,
                max_travel_mm, target_offset_m, tcp_offset_m, max_j5_deg=60.,
                max_tilt_deg=25.):
    """Same keys and checks as the grasp leg's diagnose(), minus its fixed-pose rule.

    The placement leg detects the frame after a deliberate base-frame shift, so
    "must currently sit at the fixed grasp pose" does not apply; IK limits,
    per-joint bounds, travel bound and the path checks still do. The target
    attitude is the closest to straight down that J5 can hold (see
    ``placement_attitude``), not necessarily straight down.
    """
    if not np.isfinite(max_travel_mm) or not 0 < max_travel_mm <= 400:
        raise ValueError('法兰位移上限必须在 (0,400] mm')
    joints = np.asarray(joints, float)
    vertical = np.diag([-1., 1., -1.])
    target, base = hover_target(flange, camera_point, handeye, vertical, clearance,
                                target_offset_m, tcp_offset_m)
    tcp_wanted = np.asarray(base, float) + [0., 0., clearance] + np.asarray(target_offset_m, float)
    target, attitude_joints, tilt_deg = _tilted_target(
        tcp_wanted, flange, joints, limits, fk, max_delta_deg, tcp_offset_m,
        max_j5_deg, max_tilt_deg)
    tcp = np.eye(4)
    tcp[:3, 3] = np.asarray(tcp_offset_m, float)
    flange_pose = matrix_pose(flange)
    issues = []
    if np.linalg.norm(target[:3, 3]-np.asarray(flange, float)[:3, 3]) > max_travel_mm/1000:
        issues.append(f'目标法兰位移超过{max_travel_mm:g} mm')
    report = dict(mode='DRY_RUN_ONLY', motion_enabled=False,
                  max_hover_joint_delta_deg=max_delta_deg, max_flange_travel_mm=max_travel_mm,
                  current_joints_deg=np.rad2deg(joints).tolist(), current_joints_rad=joints.tolist(),
                  current_CAN_flange_m_rad=flange_pose,
                  current_CAN_flange_mm_deg=np.r_[np.array(flange_pose[:3])*1000,
                                                  np.rad2deg(flange_pose[3:])].tolist(),
                  red_block_base_xyz_m=np.asarray(base).tolist(),
                  T_base_flange=np.asarray(flange).tolist(),
                  T_flange_camera=np.asarray(handeye).tolist(),
                  provisional_T_flange_tcp=tcp.tolist(),
                  hypothetical_target_flange_m_rad=matrix_pose(target),
                  tool_tilt_deg=tilt_deg, max_j5_deg=max_j5_deg,
                  clearance_above_detected_surface_mm=clearance*1000,
                  target_offset_mm=(np.asarray(target_offset_m)*1000).tolist(),
                  tcp_offset_mm=(np.asarray(tcp_offset_m)*1000).tolist(), issues=issues)
    try:
        q = solve_target(target, attitude_joints, limits, fk, max_delta_deg)
        report.update(IK_joints_deg=np.rad2deg(q).tolist(), IK_joints_rad=q.tolist(),
                      IK_delta_from_current_deg=np.rad2deg(q-joints).tolist())
        if max(abs(q-joints)) > np.deg2rad(max_delta_deg):
            report['issues'].append(f'IK 相对当前位置累计变化超过{max_delta_deg:g}°')
    except ValueError as error:
        report.update(IK_joints_deg=None, IK_error=str(error))
        report['issues'].append(str(error))
    return report


def _tilted_target(tcp_wanted, flange, joints, limits, fk, max_delta_deg, tcp_offset_m,
                   max_j5_deg, max_tilt_deg):
    """Target flange pose at ``tcp_wanted`` with the least tilt J5 allows."""
    rotation, attitude_joints, tilt_deg = placement_attitude(
        tcp_wanted, lean_azimuth(flange), joints, limits, fk, max_delta_deg,
        tcp_offset_m, max_j5_deg, max_tilt_deg)
    return flange_for_tcp(tcp_wanted, rotation, tcp_offset_m), attitude_joints, tilt_deg


def wait_retry_quit(controller, fd, preview, monitor, event, *, exit_event, retry_event,
                    prompt, retry_fields=None):
    """Wait for `retry` / `quit` after a recoverable rejection.

    Returns True for a fresh detection attempt and False to leave the session with
    the arm still enabled and holding. Stop keys keep requesting the stop, and the
    guard monitoring keeps running through ``monitor`` while the operator decides.
    """
    def poll():
        monitor()
        preview.pump()

    first = prompt
    while True:
        kind, typed = type_word(fd, poll, first)
        first = None
        if kind == 'stop':
            raise KeyboardInterrupt('重试等待期间用户急停')
        if kind == 'eof':
            event(dict(event=exit_event, reason='terminal_eof'))
            return False
        if typed == 'quit':
            event(dict(event=exit_event, reason='quit'))
            return False
        if typed == 'retry':
            event(dict(event=retry_event, **(retry_fields or {})))
            return True
        print(prompt, flush=True)


class LiveFrameWatch:
    """Keep detecting the black frame while a pre-planned placement path runs.

    The operator does not move the frame, so the path planned from the fixed
    grasp pose is executed unchanged: this watcher never re-plans, never edits
    the trajectory and never stops the motion (the CAN/position guards stay in
    charge of that). It only keeps the camera detection alive during the motion
    and records what the camera saw: rate-limited ``frame_live`` events with the
    detected base point and one ``frame_live_lost`` event per loss episode. It
    never prints per sample -- the terminal only gets the one-line "live
    detection on" notice and the end-of-motion summary.

    ``delta_from_plan_mm`` is the detected point minus the planned one, so it
    also carries the uncalibrated hand-eye/pose drift measured during motion
    (up to ~25 mm in the 2026-09-17 audit); it is a record, not a control input.
    """
    def __init__(self, planned_base_m, handeye, locate, event, *, period=.5,
                 now=time.monotonic):
        self.planned = np.asarray(planned_base_m, float)
        self.handeye = np.asarray(handeye, float)
        self.locate = locate
        self.event = event
        self.period, self.now = period, now
        self.last_try = float('-inf')
        self.last_stamp = None
        self.seen = 0
        self.lost = 0
        self.lost_reported = False
        self.points = []

    def update(self, preview, read_state):
        now = self.now()
        if now-self.last_try < self.period:
            return
        self.last_try = now
        sample = preview.sample()
        if sample is None or sample[0] == self.last_stamp:
            return
        self.last_stamp = sample[0]
        point = black_frame_point(sample[2], sample[1], self.locate)
        if not point['valid']:
            self.lost += 1
            if not self.lost_reported:
                self.lost_reported = True
                reason = point.get('detector_reason') or point.get('reason')
                self.event(dict(event='frame_live_lost', reason=reason))
            return
        self.lost_reported = False
        _, flange, _ = read_state()
        camera = np.asarray(point['xyz_camera_m'], float)
        base = (np.asarray(flange, float) @ self.handeye @ np.r_[camera, 1.])[:3]
        delta = (base-self.planned)*1000
        self.seen += 1
        self.points.append(base.copy())
        self.event(dict(event='frame_live', base_xyz_m=base.tolist(),
                        delta_from_plan_mm=delta.tolist(),
                        frame_candidate=point.get('frame_candidate')))

    def summary(self):
        if not self.points:
            return f'实时识别汇总：全程 {self.lost} 帧均未识别到黑框（未影响路径）。'
        points = np.asarray(self.points)
        spread = np.ptp(points, axis=0)*1000
        return (f'实时识别汇总：有效 {self.seen} 帧 / 未识别 {self.lost} 帧，'
                f'基座点波动 X {spread[0]:.1f}、Y {spread[1]:.1f}、Z {spread[2]:.1f} mm'
                f'（含未标定的手眼/位姿漂移，仅供参考）；路径未做改动。')


def observe_black_frame(controller, fd, preview, read_state, locate, event, hold_tolerance_deg,
                        frames=PLACE_OBSERVATION_FRAMES):
    """Recoverable target rejection at the fixed pose; same policy as the grasp leg.

    The arm is parked here, so several fresh frames are averaged (median) instead
    of trusting one: the box-centre estimate scatters by a few millimetres run to
    run (2026-09-18 runs 14:00/14:21/14:37/14:41: 413.8-415.8 mm in X and
    91.0-94.6 mm in Y for the same stationary frame), which is exactly the small
    placement offset the operator sees. A fixed trim cannot remove that scatter;
    averaging the observation can.
    """
    def monitor():
        controller.tick()
        if controller.locked:
            raise RuntimeError('固定点识别监控已停止')
        return read_state()

    def poll():
        poll_key(fd)
        monitor()

    attempt = 0
    while True:
        attempt += 1
        preview.stage = 'Detecting black frame at fixed pose'
        preview.detail = ''
        requested = time.monotonic()
        while True:
            poll()
            preview.pump()
            sample = preview.sample()
            if sample is not None and sample[0] >= requested:
                break
            time.sleep(.02)
        state, flange, _ = monitor()
        point = black_frame_point(sample[2], sample[1], locate)
        if point['valid']:
            collected = [np.asarray(point['xyz_camera_m'], float)]
            deadline = time.monotonic() + PLACE_OBSERVATION_TIMEOUT_S
            while len(collected) < frames and time.monotonic() < deadline:
                poll()
                preview.pump()
                newer = preview.sample()
                if newer is None or newer[0] <= sample[0]:
                    time.sleep(.02)
                    continue
                extra = black_frame_point(newer[2], newer[1], locate)
                if not extra['valid']:
                    continue
                sample, point = newer, extra
                collected.append(np.asarray(extra['xyz_camera_m'], float))
            if len(collected) > 1:
                stacked = np.asarray(collected)
                point = dict(point)
                point['xyz_camera_m'] = np.median(stacked, axis=0).tolist()
                point['observation_frames'] = len(collected)
                point['observation_spread_mm'] = float(np.max(np.ptp(stacked, axis=0))*1000)
            return sample, state, flange, point
        event(dict(event='observation_rejected', attempt=attempt,
                   observation_stamp=sample[0], detail=point))
        preview.pause_inference(sample)
        preview.wait_until_paused(poll=poll)
        preview.stage = 'Detection rejected - terminal: retry / quit'
        preview.detail = str(point.get('detector_reason') or point.get('reason'))
        print(f'识别未通过：{point}。机械臂保持固定点，未发送接近目标。\n'
              '调整黑色框/光照后输入 retry 回车重新识别；quit 退出并保持位置；空格/Esc/Ctrl+C 急停。', flush=True)
        if not wait_retry_quit(controller, fd, preview, monitor, event,
                               exit_event='observation_exit', retry_event='observation_retry',
                               prompt='请输入 retry 或 quit；尚未获得有效目标，go 不会执行运动。',
                               retry_fields=dict(attempt=attempt + 1)):
            return None
        preview.resume_inference()


def build_parser():
    """Command line parser (kept separate so the defaults stay testable)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='can1')
    parser.add_argument('--clearance-mm', type=float, default=55,
                        help='TCP 在黑色框平面上方停止高度，默认55（2026-09-14 由45增加5 mm，2026-09-18 再增加5 mm）；范围10–150')
    parser.add_argument('--max-joint-delta-deg', type=float, default=120,
                        help='相对当前姿态允许的最大单关节变化，默认120（放置要跨到垫子上，比抓取腿宽）；范围(0,170]')
    parser.add_argument('--max-travel-mm', type=float, default=400,
                        help='法兰总位移上限，默认400（放置腿从固定点跨到垫子，比抓取腿的200宽）；范围(0,400]')
    parser.add_argument('--max-j5-deg', type=float, default=60,
                        help='J5 规划上限(°)，默认60；这台机械臂 J5 在约67°就顶住（伺服12 A不动），'
                             '竖直姿态在该落点需要约87°，因此最终姿态取"J5上限内最接近竖直"；范围(0,89]')
    parser.add_argument('--max-tilt-deg', type=float, default=25,
                        help='允许的最大工具轴倾斜(°)，默认25；J5上限内凑不出小于该值的倾斜时直接拒绝规划；范围(0,60]')
    parser.add_argument('--open-gripper', action='store_true',
                        help='到位后松开夹爪（放置流程用）：按步张开、校验开度，然后沿基座 +Z 抬离；'
                             '不带该参数时这条腿从不发送夹爪指令')
    parser.add_argument('--open-mm', type=float, default=None,
                        help='松爪目标开度(mm)，默认不指定=设备最大行程；范围(0,100]')
    parser.add_argument('--open-force-n', type=float, default=2.,
                        help='松爪力参数(N)，默认2；必须是你现场验证过的值；范围(0,50]')
    parser.add_argument('--open-step-mm', type=float, default=2.,
                        help='松爪步长(mm)，默认2；范围(0.2,10]')
    parser.add_argument('--open-step-ms', type=float, default=200.,
                        help='松爪每步等待(ms)，默认200；范围50–1000')
    parser.add_argument('--retract-mm', type=float, default=30.,
                        help='松爪后沿基座 +Z 抬离高度(mm)，默认30（0=不抬离）；范围[0,150]')
    parser.add_argument('--hold-tolerance-deg', type=float, default=HOLD_TOLERANCE_DEG,
                        help='停在固定点等待时允许的关节偏离，默认0.8')
    parser.add_argument('--pre-shift-mm', type=float, nargs=3, default=[0., 0., 0.],
                        metavar=('SX', 'SY', 'SZ'),
                        help='识别前把 TCP 在基座系预移动(mm)，默认 0 0 0 即不预移动、直接在固定抓取点识别；'
                             '需要把垫子挪进视野时再给，例如 0 80 0')
    parser.add_argument('--target-base-mm', type=float, nargs=3, default=None, metavar=('X', 'Y', 'Z'),
                        help='一次性目标：基座系 TCP 绝对坐标(mm)；给出后跳过预移动与识别，直接保持姿态运动到该点')
    parser.add_argument('--target-delta-mm', type=float, nargs=3, default=None, metavar=('DX', 'DY', 'DZ'),
                        help='一次性目标：相对固定点 TCP 的基座系位移(mm)；与 --target-base-mm 二选一')
    parser.add_argument('--target-rpy-deg', type=float, nargs=3, default=None, metavar=('R', 'P', 'Y'),
                        help='一次性目标的 TCP 姿态(基座系 RPY，度；与 pose_matrix 同一约定)；默认竖直向下，'
                             '默认夹爪竖直向下 0 180 0，仅允许工具轴向下的姿态')
    parser.add_argument('--confirm-timeout-s', type=float, default=120.,
                        help='等待 go 的确认时限(秒)，默认120；超时自动重新识别而不是急停')
    parser.add_argument('--path-speed-deg-s', type=float, default=7.,
                        help='直线路径的目标速度(°/s)，默认7（2026-09-18 由5提速，固件速率同步 3%%→5%%）；范围 0.5–8')
    parser.add_argument('--final-speed-deg-s', type=float, default=5.,
                        help='下降段的速度(°/s)，默认5（2026-09-18 由4提速）；范围 0.5–8，'
                             '且不大于 --path-speed-deg-s')
    parser.add_argument('--follow-error-deg', type=float, default=1.2,
                        help='连续运动允许的反馈-目标偏差(°)，默认1.2（原审计值0.8；'
                             '4°/s 档实测最大 0.33°）；范围 0.3–2，硬界限 2°')
    parser.add_argument('--motion-envelope-deg', type=float, default=.8,
                        help='运动中的位置包络(°)，默认0.8（原审计值0.3；与上面的速度配套）；范围 0.1–2，'
                             '包络外再超 1.5° 单帧即停')
    parser.add_argument('--target-offset-mm', type=float, nargs=3, default=[5., 5., 0.],
                        metavar=('DX', 'DY', 'DZ'), help='基座系目标微调(mm)，默认 5 5 0（沿基座 X、Y 正方向各补偿5 mm）')
    parser.add_argument('--tcp-offset-mm', type=float, nargs=3, default=None,
                        metavar=('TX', 'TY', 'TZ'), help='法兰系 TCP 偏移(mm)，默认 0 0 80')
    parser.add_argument('--imgsz', type=int, choices=(320, 416, 512, 640), default=640)
    parser.add_argument('--device', default='0', help='0=GPU（默认），cpu=纯CPU')
    parser.add_argument('--no-window', action='store_true', help='不打开相机窗口')
    parser.add_argument('--min-available-mb', type=float, default=2000)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        parser.error('需要交互终端输入 go 和接收停止键')
    if not np.isfinite(args.clearance_mm) or not 10 <= args.clearance_mm <= 150:
        parser.error('--clearance-mm 必须在 10–150 mm')
    if not np.isfinite(args.max_joint_delta_deg) or not 0 < args.max_joint_delta_deg <= 170:
        parser.error('--max-joint-delta-deg 必须在 (0,170]°')
    if not np.isfinite(args.max_travel_mm) or not 0 < args.max_travel_mm <= 400:
        parser.error('--max-travel-mm 必须在 (0,400] mm')
    if not np.isfinite(args.max_j5_deg) or not 0 < args.max_j5_deg <= 89:
        parser.error('--max-j5-deg 必须在 (0,89]°')
    if not np.isfinite(args.max_tilt_deg) or not 0 < args.max_tilt_deg <= 60:
        parser.error('--max-tilt-deg 必须在 (0,60]°')
    if not np.isfinite(args.open_force_n) or not 0 < args.open_force_n <= 50:
        parser.error('--open-force-n 必须在 (0,50] N')
    if args.open_mm is not None and (not np.isfinite(args.open_mm) or not 0 < args.open_mm <= 100):
        parser.error('--open-mm 必须在 (0,100] mm')
    if not np.isfinite(args.open_step_mm) or not .2 <= args.open_step_mm <= 10:
        parser.error('--open-step-mm 必须在 0.2–10 mm')
    if not np.isfinite(args.open_step_ms) or not 50 <= args.open_step_ms <= 1000:
        parser.error('--open-step-ms 必须在 50–1000 ms')
    if not np.isfinite(args.retract_mm) or not 0 <= args.retract_mm <= 150:
        parser.error('--retract-mm 必须在 0–150 mm')
    if (args.open_gripper is False
            and (args.open_mm is not None or args.retract_mm != 30.)):
        parser.error('--open-mm / --retract-mm 只在带 --open-gripper 时有意义')
    if not np.isfinite(args.hold_tolerance_deg) or not .2 <= args.hold_tolerance_deg <= 10:
        parser.error('--hold-tolerance-deg 必须在 0.2–10°')
    args.target_offset_mm = np.asarray(args.target_offset_mm, float)
    if not np.isfinite(args.target_offset_mm).all() or np.max(np.abs(args.target_offset_mm)) > 50:
        parser.error('--target-offset-mm 必须为三个有限值，每个不超过 ±50 mm')
    args.pre_shift_mm = np.asarray(args.pre_shift_mm, float)
    if not np.isfinite(args.pre_shift_mm).all() or np.max(np.abs(args.pre_shift_mm)) > 200:
        parser.error('--pre-shift-mm 必须为三个有限值，每个不超过 ±200 mm')
    for name in ('target_base_mm', 'target_delta_mm'):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, np.asarray(value, float))
    if not np.isfinite(args.confirm_timeout_s) or not 5 <= args.confirm_timeout_s <= 600:
        parser.error('--confirm-timeout-s 必须在 5–600 秒')
    if not np.isfinite(args.path_speed_deg_s) or not .5 <= args.path_speed_deg_s <= 8:
        parser.error('--path-speed-deg-s 必须在 0.5–8°/s')
    if (not np.isfinite(args.final_speed_deg_s) or not .5 <= args.final_speed_deg_s <= 8
            or args.final_speed_deg_s > args.path_speed_deg_s):
        parser.error('--final-speed-deg-s 必须在 0.5–8°/s 且不大于 --path-speed-deg-s')
    if not np.isfinite(args.follow_error_deg) or not .3 <= args.follow_error_deg <= 2:
        parser.error('--follow-error-deg 必须在 0.3–2°')
    if not np.isfinite(args.motion_envelope_deg) or not .1 <= args.motion_envelope_deg <= 2:
        parser.error('--motion-envelope-deg 必须在 0.1–2°')
    if args.target_base_mm is not None and args.target_delta_mm is not None:
        parser.error('--target-base-mm 与 --target-delta-mm 二选一')
    for name, value, limit in (('--target-base-mm', args.target_base_mm, 1000.),
                               ('--target-delta-mm', args.target_delta_mm, 400.)):
        if value is None:
            continue
        value = np.asarray(value, float)
        if not np.isfinite(value).all() or np.max(np.abs(value)) > limit:
            parser.error(f'{name} 必须为三个有限值，每个不超过 ±{limit:g} mm')
    if args.target_rpy_deg is not None:
        if args.target_base_mm is None and args.target_delta_mm is None:
            parser.error('--target-rpy-deg 只在给出 --target-base-mm/--target-delta-mm 时有意义')
        rpy = np.asarray(args.target_rpy_deg, float)
        if not np.isfinite(rpy).all() or np.max(np.abs(rpy)) > 360:
            parser.error('--target-rpy-deg 必须为三个有限角度，每个不超过 ±360°')
        args.target_rpy_deg = rpy
    if args.tcp_offset_mm is None:
        args.tcp_offset_mm = np.asarray(TCP_OFFSET_M, float) * 1000
    args.tcp_offset_mm = np.asarray(args.tcp_offset_mm, float)
    if (not np.isfinite(args.tcp_offset_mm).all() or np.max(np.abs(args.tcp_offset_mm[:2])) > 50
            or not 0 < args.tcp_offset_mm[2] <= 200):
        parser.error('--tcp-offset-mm 横向每个不超过 ±50 mm，轴向必须在 (0,200] mm')
    one_shot = args.target_base_mm is not None or args.target_delta_mm is not None
    if not one_shot:
        memory = available_memory_mb()
        print(f'内存：可用 {memory["MemAvailable"]} MB；本入口只做识别与接近，不发送夹爪指令。', flush=True)
        if args.min_available_mb and memory['MemAvailable'] < args.min_available_mb:
            parser.error(f'可用内存 {memory["MemAvailable"]} MB 低于 {args.min_available_mb:g} MB')
    else:
        print('一次性目标模式：不使用相机、不做识别、不发送夹爪指令。', flush=True)

    from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from grasp.camera import RealSense
    from grasp.live_depth import locate
    from grasp.hover_preview import HoverPreview
    import cv2

    root = Path(__file__).resolve().parents[1]
    calibration = json.loads((root/'data/handeye/d405_02_fixed_split/candidate_result.json').read_text())
    if calibration['camera_serial'] != hrb.CAMERA_SERIAL:
        raise ValueError('手眼相机身份不匹配（如需换相机，设置环境变量 GRASP_CAMERA_SERIAL）')
    handeye = transform(calibration['T_flange_wrist'])
    model = BlackFrameModel(cv2=cv2)
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    limits = piper_planning_limits(ROBOT_JOINT_LIMIT_PRESET_RAD)
    # This arm's J5 binds a little under 70° although the firmware allows ±89°;
    # placement *plans* stay inside the real travel so the joint is never pushed
    # into that bound (2026-09-14: 12 A, no motion). The controller keeps the
    # firmware table -- its feedback check has no tolerance and must not fire on a
    # joint-frame glitch at the cap (that stopped `home` on 2026-09-14).
    plan_limits = cap_plan_limits(limits, args.max_j5_deg)
    print(f'运动参数：平移 {args.path_speed_deg_s:g}°/s、下降 {args.final_speed_deg_s:g}°/s（关节速度峰值）；'
          f'运动包络 {args.motion_envelope_deg:g}°、跟随误差 {args.follow_error_deg:g}°。', flush=True)
    print(f'J5 规划上限 ±{args.max_j5_deg:g}°（机械临界约 67°），允许工具轴倾斜 ≤{args.max_tilt_deg:g}°；'
          f'控制器仍按固件限位（J5 ±{np.rad2deg(limits[4][1]):.0f}°）。',
          flush=True)
    output = root/'yolo_grasp/runs'/('place_frame_'+time.strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    stream = dict(width=640, height=480, fps=30)
    cfg = dict(serial=hrb.CAMERA_SERIAL, color=stream, depth=stream, warmup_frames=30, timeout_ms=1000)
    preview = None
    if not one_shot:
        preview = HoverPreview(lambda: RealSense(cfg), model, cv2, args.imgsz,
                               args.device, not args.no_window)
        preview.title = 'D405 black frame | terminal: go | Q / Esc / Space: STOP'
        print(f'启动后自动到固定点；识别蓝色气垫上的黑色框，停在框上方 {args.clearance_mm:g} mm。', flush=True)
    else:
        print('启动后自动到固定点，然后保持固定姿态直接运动到给定 TCP 目标。', flush=True)
    arm = receiver = controller = trace = None
    def create(profile):
        return AgxArmFactory.create_arm(create_agx_arm_config(robot='piper_x',
            firmeware_version=profile, interface='socketcan', channel=args.channel))
    with (output/'events.jsonl').open('x', buffering=1) as log, ac.shutdown_signals():
        def event(row):
            log.write(json.dumps(dict(host_s=time.time(), **row))+'\n')
        try:
            if preview is not None:
                preview.start()
                while preview.sample() is None:
                    preview.pump()
                    time.sleep(.02)
                preview.pump()
            trace = CanTrace(args.channel)
            try:
                trace.start()
            except Exception as error:
                trace = None
                print(f'注意：双通道 CAN 记录未启动（{error}）；控制流程继续。', flush=True)
                event(dict(event='can_trace_unavailable', error=str(error)))
            arm = create('default'); arm.connect()
            firmware = query_firmware(arm); arm.disconnect()
            arm = create(resolve_firmware_profile('piper_x', firmware['software_version']))
            receiver = Receiver(args.channel); arm.connect(); time.sleep(.5)
            controller = GuardedController(arm, receiver, root/'yolo_grasp/local/arm_home.json',
                                           {}, limits, speed=3, timeout=30)
            controller.hold_tolerance_deg = args.hold_tolerance_deg
            controller.log = event
            with ac.keyboard() as fd:
                # Set only while the pre-planned placement path runs (see
                # LiveFrameWatch); None the rest of the time.
                live_watch = {'frame': None}
                def tick():
                    poll_key(fd); controller.tick()
                    if preview is not None:
                        preview.pump()
                        if live_watch['frame'] is not None:
                            live_watch['frame'].update(
                                preview, lambda: snapshot(receiver, time.time()))
                    if controller.locked:
                        raise RuntimeError('会话已停止')
                def finish():
                    while controller.active:
                        tick(); time.sleep(.02)
                if preview is not None:
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
                if args.target_base_mm is not None or args.target_delta_mm is not None:
                    # One-shot target: no pre-shift, no detection. Useful once the
                    # mat's rectangle has been measured (5 samples agreed within ~2 mm
                    # on 2026-09-11: TCP delta +110.6 / +152.9 / -92.2 mm).
                    offset = np.asarray(args.tcp_offset_mm, float)/1000
                    state, flange, _ = snapshot(receiver, time.time())
                    if args.target_rpy_deg is None:
                        # Same rule as the detection flow: straight down is out of
                        # J5's real travel here, so aim for the closest tilt it can
                        # hold instead of demanding 0°.
                        _, tcp_wanted = tcp_target_pose(flange, offset,
                                                        base_mm=args.target_base_mm,
                                                        delta_mm=args.target_delta_mm)
                        target, _, tilt_deg = _tilted_target(
                            tcp_wanted, flange, state.joints, plan_limits, fk,
                            args.max_joint_delta_deg, offset, args.max_j5_deg,
                            args.max_tilt_deg)
                        print(f'最终姿态：J5 上限 {args.max_j5_deg:g}° → 工具轴与竖直 {tilt_deg:.1f}°'
                              f'（竖直姿态在该点需要超过 J5 机械行程，已按上限取最小倾斜）。', flush=True)
                    else:
                        target, tcp_wanted = tcp_target_pose(flange, offset,
                                                             base_mm=args.target_base_mm,
                                                             delta_mm=args.target_delta_mm,
                                                             rpy_deg=args.target_rpy_deg)
                    if np.any(args.target_offset_mm):
                        # Same empirical trim as the detection flow, so both paths
                        # share one correction.
                        trim = np.asarray(args.target_offset_mm, float)/1000
                        tcp_wanted = tcp_wanted + trim
                        target[:3, 3] = tcp_wanted - target[:3, :3] @ offset
                        print(f'（叠加目标微调 {np.round(args.target_offset_mm,1).tolist()} mm）', flush=True)
                    tcp_now = flange[:3, 3] + flange[:3, :3] @ offset
                    print(f'一次性目标：TCP 从 {np.round(tcp_now*1000,1).tolist()} 到 '
                          f'{np.round(tcp_wanted*1000,1).tolist()} mm（基座系，最终工具轴竖直向下）；'
                          f'位移 {np.linalg.norm(tcp_wanted-tcp_now)*1000:.1f} mm。', flush=True)
                    tool_axis = target[:3, 2]
                    print(f'目标姿态 RPY(°)={np.round(np.rad2deg(matrix_pose(target)[3:]),1).tolist()}，'
                          f'工具轴与竖直夹角 {np.rad2deg(np.arccos(np.clip(abs(tool_axis[2]),-1,1))):.1f}°'
                          f'{"（竖直向下）" if abs(tool_axis[2]) > .999 else ""}。', flush=True)
                    event(dict(event='target_requested',
                               tcp_now_mm=(tcp_now*1000).tolist(),
                               tcp_target_mm=(tcp_wanted*1000).tolist(),
                               target_flange_m_rad=matrix_pose(target)))
                    # Explicit waypoints on the straight line: the firmware's own
                    # joint-space move is not a straight joint interpolation, and on
                    # 2026-09-11 it dipped below the frame and swung back up.
                    approach, descent = vertical_paths(flange, target, state.joints, plan_limits,
                                                       fk, args.max_joint_delta_deg, offset,
                                                       args.max_travel_mm,
                                                       max_tilt_deg=args.max_tilt_deg)
                    path = approach + descent[1:]
                    step = np.rad2deg(np.max(np.abs(np.diff(path, axis=0))))
                    segments = [(approach, args.path_speed_deg_s, '平移到正上方'),
                                (descent, args.final_speed_deg_s, '转向下降+垂直下沉')]
                    total = 0.
                    prepared_segments = []
                    for index, (segment, speed, label) in enumerate(segments):
                        times, targets = timed_path(np.asarray(segment, float), speed)
                        prepared_segments.append((times, targets))
                        total += float(times[-1])
                        event(dict(event='target_plan' if index == 0 else 'target_plan_final',
                                   segment=label, points=len(segment), speed_deg_s=speed,
                                   duration_s=float(times[-1]), max_step_deg=float(step),
                                   target_flange_m_rad=matrix_pose(target)))
                        print(f'{label} 直线路径 {len(segment)} 点、最大单步 {step:.3f}°、'
                              f'{speed:g}°/s、预计 {times[-1]:.1f} 秒'
                              f'{"（TCP 直线平移，姿态保持不变）" if index == 0 else "（下降中把工具轴转到竖直，末段垂直下沉）"}。',
                              flush=True)
                    wait_for_fresh_feedback(receiver)
                    wait_for_command(controller, fd, preview, 'go',
                                     '路径已规划。输入 go 回车执行指定坐标目标；quit 退出。')
                    current, current_pose, _ = snapshot(receiver, time.time())
                    distance, angle = pose_error(current_pose, flange)
                    tolerance_m, tolerance_rad = parked_flange_tolerance(args.hold_tolerance_deg)
                    if (np.max(abs(current.joints-state.joints)) > np.deg2rad(args.hold_tolerance_deg)
                            or distance > tolerance_m or angle > tolerance_rad):
                        raise ValueError('确认期间机械臂位置变化，目标作废')
                    for times, targets in prepared_segments:
                        wait_for_fresh_feedback(receiver)
                        execute_path(controller, times, targets, tick, event,
                                     hold_tolerance_deg=args.hold_tolerance_deg,
                                     follow_error_deg=args.follow_error_deg,
                                     motion_tolerance_deg=args.motion_envelope_deg,
                                     speed_percent=PLACE_SPEED_PERCENT)
                    print(f'两段合计预计 {total:.1f} 秒，已到达。', flush=True)
                    _, reached, _ = snapshot(receiver, time.time())
                    distance, angle = pose_error(reached, target)
                    if distance > .003 or angle > np.deg2rad(1):
                        raise ValueError(f'一次性目标法兰误差超限：{distance*1000:.1f} mm / '
                                         f'{np.rad2deg(angle):.2f}°')
                    tcp_reached = reached[:3, 3] + reached[:3, :3] @ offset
                    residual = float(np.linalg.norm(tcp_reached-tcp_wanted)*1000)
                    print(f'到位 TCP(mm)：{np.round(tcp_reached*1000,1).tolist()}；'
                          f'与目标差 {residual:.1f} mm。', flush=True)
                    event(dict(event='target_reached', tcp_mm=(tcp_reached*1000).tolist(),
                               residual_mm=residual, joint_rad=controller.state().joints.tolist()))
                    controller.guard_start = controller.state().joints.copy()
                    controller.guard_goal = controller.guard_start.copy()
                    controller.guard_tolerance_deg = args.hold_tolerance_deg
                    return hold_after_place(controller, fd, preview, event)
                if np.any(args.pre_shift_mm):
                    # The mat does not fit in the camera view from the fixed pose;
                    # shift the TCP first, then detect from there.
                    wait_for_command(controller, fd, preview, 'shift',
                        f'已到固定点。输入 shift 回车：TCP 在基座系预移动 '
                        f'{np.round(args.pre_shift_mm,1).tolist()} mm 后开始识别；quit 退出。')
                    state, flange, _ = snapshot(receiver, time.time())
                    target = shifted_flange(flange, args.pre_shift_mm)
                    # Joint-space SDK move at 3% (same as the fixed-pose leg): an
                    # 80 mm sideways shift needs ~47 deg of joint motion, which
                    # took 80 s on the 2 deg/s timed path (2026-09-11 14:47).
                    q_shift = solve_target(target, state.joints, plan_limits, fk,
                                           args.max_joint_delta_deg)
                    delta = np.rad2deg(q_shift-state.joints)
                    event(dict(event='pre_shift_plan', shift_mm=args.pre_shift_mm.tolist(),
                               joint_rad=state.joints.tolist(), target_joint_rad=q_shift.tolist()))
                    print('预移动各关节需要移动(°)：' + '，'.join(
                        f'J{i+1} {v:+.1f}' for i, v in enumerate(delta)), flush=True)
                    controller.guard_start = state.joints.copy()
                    controller.guard_goal = q_shift.copy()
                    controller.guard_tolerance_deg = MOTION_TOLERANCE_DEG
                    controller.experiment_target = q_shift.copy()
                    started = time.monotonic()
                    controller.command('home')
                    while controller.active:
                        tick(); time.sleep(.02)
                    print(f'预移动完成，用时 {time.monotonic()-started:.1f} 秒（3% 速度）。', flush=True)
                    _, reached, _ = snapshot(receiver, time.time())
                    d, a = pose_error(reached, target)
                    if d > .003 or a > np.deg2rad(1):
                        raise ValueError(f'预移动后法兰误差超限：{d*1000:.1f} mm / {np.rad2deg(a):.2f}°')
                    event(dict(event='pre_shift_reached', error_mm_deg=[d*1000, float(np.rad2deg(a))],
                               joint_rad=controller.state().joints.tolist()))
                    print(f'预移动完成（法兰误差 {d*1000:.1f} mm / {np.rad2deg(a):.2f}°），开始识别黑色框。',
                          flush=True)
                    controller.guard_start = controller.state().joints.copy()
                    controller.guard_goal = controller.guard_start.copy()
                    controller.guard_tolerance_deg = args.hold_tolerance_deg
                else:
                    print('已到固定点，正在识别黑色框。', flush=True)
                attempt = 0
                while True:
                    attempt += 1
                    observation = observe_black_frame(controller, fd, preview,
                                                      lambda: snapshot(receiver, time.time()),
                                                      locate, event, args.hold_tolerance_deg)
                    if observation is None:
                        print('识别流程已退出，机械臂保持使能和当前位置。', flush=True)
                        return 0
                    sample, state, flange, point = observation
                    observed_at, frame, prediction, annotated = sample
                    attempt_output = output/f'attempt_{attempt:03d}'
                    attempt_output.mkdir(exist_ok=False)
                    frame.save(attempt_output/'observation')
                    if not cv2.imwrite(str(attempt_output/'detection.png'), annotated):
                        raise OSError('无法保存检测画面')
                    preview.pause_inference(sample)
                    preview.wait_until_paused(poll=lambda: poll_key(fd))
                    preview.stage = 'Planning path above the black frame'
                    preview.pump()
                    # Keeps the guards and the CAN monitoring alive while a
                    # rejected plan waits for retry / quit; it must not read keys
                    # itself, or the typed word would be eaten one character at a time.
                    def planning_monitor():
                        controller.tick()
                        if controller.locked:
                            raise RuntimeError('规划重试等待已停止会话')
                    try:
                        report = plan_report(state.joints, flange, point['xyz_camera_m'], handeye,
                                             plan_limits, fk, args.clearance_mm/1000,
                                             args.max_joint_delta_deg, args.max_travel_mm,
                                             args.target_offset_mm/1000, args.tcp_offset_mm/1000,
                                             args.max_j5_deg, args.max_tilt_deg)
                        (output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
                        (attempt_output/'diagnostic.json').write_text(json.dumps(report, indent=2)+'\n')
                        print(f"黑框基座XYZ(mm)：{np.round(np.array(report['red_block_base_xyz_m'])*1000, 1)}")
                        if point.get('observation_frames'):
                            print(f"  观测取中值：{point['observation_frames']} 帧，"
                                  f"相机系波动 ≤{point.get('observation_spread_mm', 0):.1f} mm", flush=True)
                        if 'frame_candidate' in point:
                            picked = point['frame_candidate']
                            print(f"  采用候选：框 {[round(v) for v in picked['box_xyxy']]}、"
                                  f"深度 {picked['depth_m']:.3f} m、尺寸误差 {picked['size_error']*100:.0f}%"
                                  f"（已知矩形 70×60 mm，理论 {picked['expected_px'][0]:.0f}×"
                                  f"{picked['expected_px'][1]:.0f} px）", flush=True)
                        print(f'目标：黑框上方 {args.clearance_mm:g} mm；'
                              f'J5 上限 {args.max_j5_deg:g}° → 工具轴与竖直 '
                              f'{report.get("tool_tilt_deg", float("nan")):.1f}°；'
                              f'TCP 偏移 {np.round(args.tcp_offset_mm, 1).tolist()} mm（法兰系）；'
                              f'单关节变化上限 {args.max_joint_delta_deg:g}°。')
                        if report['issues']:
                            raise ValueError('放置规划拒绝：' + '；'.join(report['issues']))
                        approach, descent = vertical_paths(
                            flange, pose_matrix(report['hypothetical_target_flange_m_rad']),
                            state.joints, plan_limits, fk, args.max_joint_delta_deg,
                            args.tcp_offset_mm/1000, args.max_travel_mm,
                            max_tilt_deg=args.max_tilt_deg)
                        path = approach + descent[1:]
                        prepared_segments = [timed_path(approach, args.path_speed_deg_s),
                                             timed_path(descent, args.final_speed_deg_s)]
                        duration = sum(t[-1] for t, _ in prepared_segments)
                        step = np.rad2deg(np.max(np.abs(np.diff(path, axis=0))))
                        print(f'路径 {len(path)} 点、最大单步 {step:.3f}°、'
                              f'水平段 {args.path_speed_deg_s:g}°/s、下降段 {args.final_speed_deg_s:g}°/s'
                              f'（固件速率 {PLACE_SPEED_PERCENT}%）；'
                              f'先平移到黑框正上方、下降中把工具轴转到目标姿态、最后保持姿态垂直下沉，'
                              f'预计 {duration:.1f} 秒，终点按法兰 3 mm / 1° 复核。', flush=True)
                    except ValueError as error:
                        # A rejected plan is not a fault: nothing has been sent and
                        # the observation may simply not be placeable (2026-09-14 the
                        # tool-down attitude was asked for too high up and the IK ran
                        # into the J5 limit). Re-detect instead of stopping the arm.
                        event(dict(event='planning_rejected', error=str(error),
                                   attempt_dir=attempt_output.name))
                        preview.stage = 'Planning rejected - terminal: retry / quit'
                        preview.detail = str(error)
                        print(f'规划未通过：{error}。机械臂保持当前位置，未发送任何接近目标。\n'
                              '调整黑色框/垫子后输入 retry 回车重新识别；quit 退出并保持位置；'
                              '空格/Esc/Ctrl+C 急停。', flush=True)
                        if not wait_retry_quit(controller, fd, preview, planning_monitor, event,
                                               exit_event='planning_exit',
                                               retry_event='planning_retry', retry_fields=dict(
                                                   attempt=attempt + 1),
                                               prompt='请输入 retry 或 quit；尚未发送任何接近目标。'):
                            print('识别流程已退出，机械臂保持使能和当前位置。', flush=True)
                            return 0
                        preview.resume_inference()
                        continue
                    event(dict(event='plan', diagnostic=report, attempt_dir=attempt_output.name,
                               black_frame_candidate=point.get('frame_candidate'),
                               joint_path_rad=[q.tolist() for q in path]))
                    wait_for_fresh_feedback(receiver)
                    preview.stage = 'Waiting for go in TERMINAL'
                    if wait_go_frame(controller, fd, observed_at, preview, args.confirm_timeout_s):
                        break
                    # Expired: discard this observation and detect again. Nothing
                    # has been sent, so the session keeps running.
                    preview.resume_inference()
                current, current_pose, _ = snapshot(receiver, time.time())
                d, a = pose_error(current_pose, flange)
                flange_tolerance_m, flange_tolerance_rad = parked_flange_tolerance(args.hold_tolerance_deg)
                if (np.max(abs(current.joints-state.joints)) > np.deg2rad(args.hold_tolerance_deg)
                        or d > flange_tolerance_m or a > flange_tolerance_rad):
                    raise ValueError('确认期间机械臂位置变化，目标作废')
                # The path is fixed from here: keep detecting the black frame in
                # real time for the operator, but never re-plan or re-target it.
                preview.set_detection_interval(LIVE_DETECTION_INTERVAL_S)
                preview.resume_inference()
                preview.stage = 'Continuous approach above the black frame (live frame detection)'
                observer = LiveFrameWatch(report['red_block_base_xyz_m'], handeye, locate, event)
                live_watch['frame'] = observer
                print('实时识别已开启：接近过程中持续检测黑框，但**不重规划**，'
                      '按固定抓取点规划好的路径执行。', flush=True)
                for trajectory_times, trajectory_targets in prepared_segments:
                    wait_for_fresh_feedback(receiver)
                    execute_path(controller, trajectory_times, trajectory_targets, tick, event,
                                 hold_tolerance_deg=args.hold_tolerance_deg,
                                 follow_error_deg=args.follow_error_deg,
                                 motion_tolerance_deg=args.motion_envelope_deg,
                                 speed_percent=PLACE_SPEED_PERCENT)
                live_watch['frame'] = None
                preview.set_detection_interval(0.)
                print(observer.summary(), flush=True)
                # Release and retract do not need the detector; pause it again so
                # the gripper and the short retract path keep the CPU to themselves.
                preview.pause_inference(preview.sample())
                preview.wait_until_paused(poll=lambda: poll_key(fd))
                preview.stage = 'Above the black frame - releasing'
                _, reached, _ = snapshot(receiver, time.time())
                d, a = pose_error(reached, pose_matrix(report['hypothetical_target_flange_m_rad']))
                if d > .003 or a > np.deg2rad(1):
                    raise ValueError('上方目标最终反馈误差超限')
                print(f'已停在黑框上方（法兰误差 {d*1000:.1f} mm / {np.rad2deg(a):.2f}°）。', flush=True)
                event(dict(event='place_hover_reached', error_mm_deg=[d*1000, float(np.rad2deg(a))],
                           joint_rad=controller.state().joints.tolist()))
                if args.open_gripper:
                    preview.stage = 'Releasing the block'
                    release_block(controller, arm, receiver, args, tick, event, plan_limits,
                                  fk, lambda receiver_, now: snapshot(receiver_, now),
                                  args.tcp_offset_mm/1000)
                return hold_after_place(controller, fd, preview, event)
        except UserQuit:
            event(dict(event='exit_requested', reason='quit_before_motion'))
            print('退出，机械臂保持使能和当前固定点位置。', flush=True)
            return 0
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
            if trace is not None: trace.close()
            if receiver is not None: receiver.close()
            if arm is not None: arm.disconnect()
            if preview is not None: preview.close()


def hold_after_place(controller, fd, preview, event):
    """Indefinite hold: `home` returns through the fixed grasp pose, `quit` exits."""
    event(dict(event='hold_started', joint_rad=controller.state().joints.tolist()))
    print('已到位并保持：输入 home 返回（先回固定抓取点再回初始位置）；'
          'quit 退出并保持位置；空格/Esc/Ctrl+C 急停。', flush=True)
    if preview is not None:
        preview.stage = 'Holding position - terminal: home / quit'

    def poll():
        controller.tick()
        if preview is not None:
            preview.pump()
        if controller.locked:
            raise RuntimeError('保持监控已停止会话')

    def home_tick():
        poll_key(fd); controller.tick()
        if preview is not None:
            preview.pump()

    while True:
        kind, typed = type_word(fd, poll)
        if kind == 'stop':
            raise KeyboardInterrupt('到位保持期间用户急停')
        if kind == 'eof':
            event(dict(event='hold_exit', reason='terminal_eof', enabled_hold=True))
            return 0
        if typed == 'quit':
            event(dict(event='hold_exit', reason='quit', enabled_hold=True))
            print('退出监控，机械臂保持使能和当前位置。', flush=True)
            return 0
        if typed == 'home':
            event(dict(event='home_requested', joint_rad=controller.state().joints.tolist()))
            go_home_via_fixed_pose(controller, home_tick)
            controller.guard_start = controller.state().joints.copy()
            controller.guard_goal = controller.guard_start.copy()
            controller.guard_tolerance_deg = controller.hold_tolerance_deg
            event(dict(event='home_reached', joint_rad=controller.state().joints.tolist()))
            print('已先回固定抓取点、再回初始位置，继续保持；输入 quit 退出。', flush=True)
        else:
            print('请输入 home 或 quit（Backspace/Delete 可改）；空格/Esc/Ctrl+C 急停。', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
