"""Timed CAN/J targets with bounded increments and feedback monitoring."""
import time
import numpy as np


# Guard tolerances, in degrees.
# The arm reproducibly sits still and then snaps 0.3-0.4 deg about 12-16 s
# after a move finishes (2026-09-10 hold record and 2026-09-11 runs). A
# commanded move keeps the tight envelope, because a large excursion there is
# a genuine following error; a parked hold must tolerate the documented snap
# instead of requesting an emergency stop.
#
# The hold default stays at or below the 0.8 deg following-error bound enforced
# while targets are sent, so a parked offset that was tolerated can never make
# the next continuous motion refuse to start.
MOTION_TOLERANCE_DEG = .3
HOLD_TOLERANCE_DEG = .8

# The joint-state frames 0x2A5-0x2A7 occasionally report a single wrong sample
# of about 0.1-0.4 deg (documented zero-zone record in BASE_AXES_TEST.md; the
# 2026-09-11 10:32 run aborted on one -0.369 deg J2 sample during motion).
# Guards therefore stop only when the excursion persists for
# GUARD_PERSIST_SAMPLES consecutive checks, and immediately beyond
# GUARD_HARD_MARGIN_DEG, so a real following error still stops the session.
GUARD_PERSIST_SAMPLES = 3
GUARD_HARD_MARGIN_DEG = 1.5
FOLLOW_ERROR_DEG = .8
FOLLOW_HARD_DEG = 2.0

# Guards must not act on readings the joint frames are known to get wrong:
# * inside the documented zero zone (|q| <= ZERO_ZONE_DEG) the joint frame can
#   report exactly 0.000 while the motor frame shows up to ~0.4 deg (verified
#   2026-09-11 11:30: joint frame 0.000, motor frame -0.286, velocity 0);
# * excursions of a few counts (1 count = 0.001 deg) are quantisation noise.
GUARD_NOISE_FLOOR_DEG = .05
ZERO_ZONE_DEG = .35
ZERO_ZONE_MAX_ERROR_DEG = .7


class TargetChanged(Exception):
    """Recoverable request to discard the remaining approach trajectory."""


def hold_current(controller, tick, event, *, now=time.monotonic, wall=time.time,
                 sleep=time.sleep, hold_tolerance_deg=HOLD_TOLERANCE_DEG):
    """Replace the old target with fresh measured joints; require 0.5 s settled feedback.

    This is a normal CAN/J position hold, not a firmware emergency stop/reset.
    Any failure still propagates to the executor's emergency-stop handler.
    """
    tick()
    state = controller.state()
    state.healthy()
    if (controller.locked or not all(state.enabled) or state.ctrl_mode != 1
            or state.mode_feedback != 1):
        raise RuntimeError('重规划保持：控制状态无效')
    goal = state.joints.copy()
    controller.validate_joints(goal)
    controller.guard_start = goal.copy()
    controller.guard_goal = goal.copy()
    controller.guard_tolerance_deg = hold_tolerance_deg
    controller.arm._msg_mode.move_spd_rate_ctrl = 3
    controller.arm.set_auto_set_motion_mode_enabled(True)
    issued = wall()
    controller.arm.move_j(goal.tolist())
    event(dict(event='tracking_hold_requested', joint_rad=goal.tolist()))
    deadline, since = now()+controller.timeout, None
    while now() < deadline:
        tick()
        state = controller.state()
        state.healthy()
        if (controller.locked or not all(state.enabled) or state.ctrl_mode != 1
                or state.mode_feedback != 1):
            raise RuntimeError('重规划停稳期间控制状态丢失')
        stationary = (np.all(state.stamps > issued) and state.motion_status == 0
                      and np.max(abs(state.joints-goal)) <= np.deg2rad(.2))
        stamp = float(np.min(state.stamps))
        since = (stamp if since is None else since) if stationary else None
        if since is not None and stamp-since >= .5:
            controller.guard_start = state.joints.copy()
            controller.guard_goal = state.joints.copy()
            event(dict(event='tracking_hold_settled', joint_rad=state.joints.tolist()))
            return
        sleep(.02)
    raise TimeoutError('重规划保持目标未停稳')


def timed_path(path, speed_deg_s=2., period=.02):
    """Quintic time scaling of joint-space arc length; zero endpoint speed."""
    path = np.asarray(path, float)
    if path.ndim != 2 or path.shape[1] != 6 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError('连续路径必须包含至少两个有限六轴目标')
    # The default stays 2 deg/s (the audited approach speed); the placement leg
    # may ask for more, so the hard ceiling is higher while the default is not.
    if not np.isfinite(speed_deg_s) or not 0 < speed_deg_s <= 8:
        raise ValueError('连续目标速度必须在 (0,8]°/s')
    distances = np.max(abs(np.diff(path, axis=0)), axis=1)
    if np.max(distances) > np.deg2rad(.5):
        raise ValueError('规划路径单步超过0.5°')
    arc = np.r_[0., np.cumsum(distances)]
    keep = np.r_[True, np.diff(arc) > 1e-12]
    arc, path = arc[keep], path[keep]
    # max derivative of 10u^3-15u^4+6u^5 is 1.875.
    duration = max(2., 1.875*arc[-1]/np.deg2rad(speed_deg_s))
    times = np.linspace(0., duration, int(np.ceil(duration/period))+1)
    u = times/duration
    progress = (10*u**3-15*u**4+6*u**5)*arc[-1]
    targets = np.column_stack([np.interp(progress, arc, path[:, j]) for j in range(6)])
    return times, targets


def execute_path(controller, times, targets, tick, event, now=time.monotonic,
                 wall=time.time, sleep=time.sleep, hold_tolerance_deg=HOLD_TOLERANCE_DEG,
                 follow_error_deg=FOLLOW_ERROR_DEG, motion_tolerance_deg=MOTION_TOLERANCE_DEG,
                 monitor=None, speed_percent=3):
    """No per-waypoint arrival wait; only final fresh feedback may finish motion.

    Normal SDK move_j mode/target transactions are retained. This deliberately
    does not switch to unsmoothed JS/MIT mode. A late scheduler never catches up
    by bursting old targets; feedback loss, lag or timing failure aborts.

    ``hold_tolerance_deg`` only widens the parked-arm checks the documented
    snap can trip: the distance between the first target and the feedback at
    start, and the guard installed while waiting at the endpoint. The moving
    envelope and every following-error bound stay at their original values.
    """
    if not np.isfinite(follow_error_deg) or not 0 < follow_error_deg <= FOLLOW_HARD_DEG:
        raise ValueError('跟随误差上限必须在 (0,2]°')
    if not np.isfinite(motion_tolerance_deg) or not 0 < motion_tolerance_deg <= FOLLOW_HARD_DEG:
        raise ValueError('运动包络必须在 (0,2]°')
    times, targets = np.asarray(times, float), np.asarray(targets, float)
    if (times.ndim != 1 or len(times) < 2 or targets.shape != (len(times), 6)
            or not np.isfinite(times).all() or not np.isfinite(targets).all()
            or times[0] != 0 or np.any(np.diff(times) <= 0)
            or np.any(np.diff(times) > .021)):
        raise ValueError('连续轨迹时间或目标无效')
    increments = np.max(abs(np.diff(targets, axis=0)), axis=1)
    if np.any(increments/np.diff(times) > np.deg2rad(8.001)):
        raise ValueError('连续轨迹速度超过8°/s')
    # Late scheduler ticks must never be answered with a burst of targets. The
    # allowed send rate is the plan's own peak rate (the quintic peaks exactly at
    # the requested speed), so this holds at any --path-speed-deg-s.
    planned_rate = float(np.max(increments/np.diff(times)))
    for q in targets:
        controller.validate_joints(q)
    if controller.locked or controller.active or not controller.prepared:
        raise RuntimeError('控制会话未就绪，拒绝连续运动')
    base_tick = tick
    def tick():
        base_tick()
        if monitor is not None:
            reason = monitor()
            if reason:
                raise TargetChanged(reason)

    previous_auto = controller.arm.get_auto_set_motion_mode_enabled()
    try:
        tick()
        start = controller.state().joints.copy()
        if np.max(abs(targets[0]-start)) > np.deg2rad(hold_tolerance_deg):
            raise RuntimeError(f'连续轨迹起点与反馈相差超过{hold_tolerance_deg:g}°')
        if not isinstance(speed_percent, int) or not 1 <= speed_percent <= 100:
            raise ValueError('运动速度百分比必须是 1–100 的整数')
        controller.arm._msg_mode.move_spd_rate_ctrl = speed_percent
        controller.arm.set_auto_set_motion_mode_enabled(True)
        started = last_sent = now()
        history = [(started, start)]
        last_target = start
        issued = None
        follow_strikes = 0
        for index, (offset, target) in enumerate(zip(times, targets)):
            due = started + offset
            while now() < due:
                tick()
                sleep(min(.01, max(0., due-now())))
            tick()
            if controller.locked:
                raise RuntimeError('连续运动会话已停止')
            if now()-due > .1:
                raise TimeoutError('连续目标调度延迟超过100 ms，停止而不补发')
            state = controller.state()
            state.healthy()
            if not all(state.enabled) or state.ctrl_mode != 1 or state.mode_feedback != 1:
                raise RuntimeError('连续运动使能或 CAN/J 模式丢失')
            follow_error = float(max(np.max(abs(state.joints-last_target)),
                                     np.max(abs(target-state.joints))))
            if follow_error > np.deg2rad(FOLLOW_HARD_DEG):
                raise RuntimeError(f'连续运动反馈与目标相差{follow_error:.3f}°，超过硬界限{FOLLOW_HARD_DEG:g}°')
            if follow_error > np.deg2rad(follow_error_deg):
                follow_strikes += 1
                if follow_strikes >= GUARD_PERSIST_SAMPLES:
                    raise RuntimeError(f'连续{follow_strikes}个目标反馈与目标相差超过{follow_error_deg:g}°')
            else:
                follow_strikes = 0
            sent = now()
            if index and np.max(abs(target-last_target)) > planned_rate*(sent-last_sent)*1.001:
                raise TimeoutError('调度抖动导致目标发送过快，拒绝补发')
            while len(history) > 1 and history[1][0] < sent-.25:
                history.pop(0)
            envelope = np.array([q for _, q in history]+[target])
            controller.guard_start = envelope.min(axis=0)
            controller.guard_goal = envelope.max(axis=0)
            # The audited 0.3 deg band fits the 2 deg/s approach; faster paths
            # need a wider band for the same physical tracking lag.
            controller.guard_tolerance_deg = motion_tolerance_deg
            issued = wall()
            controller.arm.move_j(target.tolist())
            history.append((sent, target.copy()))
            last_target, last_sent = target, sent
            event(dict(event='continuous_target', index=index, joint_rad=target.tolist()))
            # Rebase the next deadline: scheduler jitter must not cause catch-up bursts.
            started += max(0., sent-due)
        deadline, since = now()+controller.timeout, None
        while now() < deadline:
            tick()
            if controller.locked:
                raise RuntimeError('最终到位监控已停止')
            state = controller.state()
            stationary = (np.all(state.stamps > issued) and state.motion_status == 0
                          and np.max(abs(state.joints-targets[-1])) <= np.deg2rad(.2))
            stamp = float(np.min(state.stamps))
            since = (stamp if since is None else since) if stationary else None
            if since is not None and stamp-since >= .5:
                controller.guard_start = targets[-1].copy()
                controller.guard_goal = targets[-1].copy()
                controller.guard_tolerance_deg = hold_tolerance_deg
                event(dict(event='continuous_reached', joint_rad=state.joints.tolist()))
                return True
            sleep(.02)
        raise TimeoutError('连续轨迹终点未在限定时间内到位并停稳')
    except BaseException as error:
        if isinstance(error, TargetChanged):
            event(dict(event='tracking_interrupt', reason=str(error)))
            print(f'视觉监控请求停稳：{error}；取消剩余轨迹，保持当前位置。', flush=True)
            try:
                hold_current(controller, base_tick, event, now=now, wall=wall, sleep=sleep,
                             hold_tolerance_deg=hold_tolerance_deg)
                return False
            except BaseException as hold_error:
                if not controller.locked:
                    controller.stop(str(hold_error))
                raise
        if not controller.locked:
            controller.stop(str(error))
        raise
    finally:
        controller.arm.set_auto_set_motion_mode_enabled(previous_auto)
