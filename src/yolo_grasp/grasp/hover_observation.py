"""Recoverable target rejection while the robot holds the fixed pose."""
import time
import numpy as np
from .mat_pose import poll_key
from .terminal_input import type_word
from .hover_motion import (GUARD_HARD_MARGIN_DEG, GUARD_NOISE_FLOOR_DEG,
                           GUARD_PERSIST_SAMPLES, HOLD_TOLERANCE_DEG,
                           ZERO_ZONE_DEG, ZERO_ZONE_MAX_ERROR_DEG)


def _trusted_excursion(state_joints, anchor_joints, tolerance_deg):
    """Deviation beyond ``tolerance_deg`` that the joint frames can be trusted on.

    Readings inside the documented zero zone are ignored: the joint frame can
    report exactly 0.000 there while the motor frame shows up to ~0.4 deg.
    """
    state_joints = np.asarray(state_joints, float)
    deviation = np.abs(state_joints - np.asarray(anchor_joints, float))
    zero_zone = ((np.abs(state_joints) <= np.deg2rad(ZERO_ZONE_DEG))
                 & (deviation <= np.deg2rad(ZERO_ZONE_MAX_ERROR_DEG)))
    return float(np.max(np.where(zero_zone, 0., deviation))) - np.deg2rad(tolerance_deg)


def _box_values(box):
    xyxy = np.asarray(box.xyxy[0].cpu().numpy(), float)
    confidence = float(box.conf.item()) if hasattr(box, 'conf') else 1.
    return xyxy, confidence


def _box_iou(first, second):
    lo = np.maximum(first[:2], second[:2])
    hi = np.minimum(first[2:], second[2:])
    intersection = float(np.prod(np.maximum(0., hi-lo)))
    first_area = float(np.prod(np.maximum(0., first[2:]-first[:2])))
    second_area = float(np.prod(np.maximum(0., second[2:]-second[:2])))
    union = first_area+second_area-intersection
    return 0. if union <= 0 else intersection/union


def _select_red_box(boxes, previous_box_xyxy=None):
    """Suppress duplicate boxes, then conservatively associate a real multi-box frame."""
    candidates = sorted((_box_values(box) for box in boxes), key=lambda row: -row[1])
    kept = []
    for xyxy, confidence in candidates:
        if not any(_box_iou(xyxy, other[0]) >= .5 for other in kept):
            kept.append((xyxy, confidence))
    if len(kept) == 1:
        method = 'single' if len(candidates) == 1 else 'overlap_deduplicated'
        return kept[0], kept, method
    if previous_box_xyxy is None or not kept:
        return None, kept, 'ambiguous'
    previous = np.asarray(previous_box_xyxy, float)
    previous_center = (previous[:2]+previous[2:])/2
    previous_diagonal = float(np.linalg.norm(previous[2:]-previous[:2]))
    ranked = sorted((float(np.linalg.norm((xyxy[:2]+xyxy[2:])/2-previous_center)),
                     xyxy, confidence) for xyxy, confidence in kept)
    best_distance = ranked[0][0]
    association_limit = max(40., previous_diagonal)
    separation = (float('inf') if len(ranked) == 1 else ranked[1][0]-best_distance)
    if best_distance > association_limit or separation < max(10., .25*previous_diagonal):
        return None, kept, 'ambiguous'
    return (ranked[0][1], ranked[0][2]), kept, 'previous_box_association'


def target_point(sample, names, locate, previous_box_xyxy=None):
    _, frame, prediction, _ = sample
    if hasattr(frame, 'detected_point'):
        return dict(frame.detected_point)  # Validated by the isolated RGB-D worker.
    boxes = [b for b in prediction.boxes if names[int(b.cls.item())] == 'red_block']
    selected, kept, method = _select_red_box(boxes, previous_box_xyxy)
    detail = dict(count=len(boxes), deduplicated_count=len(kept), selection=method,
                  confidences=[confidence for _, confidence in kept])
    if selected is None:
        return dict(valid=False, reason='red_block_count', **detail)
    xyxy, confidence = selected
    point = dict(locate(frame, xyxy, min_depth=.07, max_depth=.6))
    point.update(detail, selected_confidence=confidence, selected_box_xyxy=xyxy.tolist())
    if not point['valid'] and point.get('pixel_uv') is not None:
        u, v = point['pixel_uv']
        if 0 <= v < frame.depth_m.shape[0] and 0 <= u < frame.depth_m.shape[1]:
            raw = float(frame.depth_m[v, u])
            point['center_raw_m'] = raw if np.isfinite(raw) else None
            point['center_depth_category'] = (
                'nonfinite' if not np.isfinite(raw) else 'zero_or_negative' if raw <= 0
                else 'below_70mm' if raw < .07 else 'above_600mm' if raw > .6 else 'in_range')
    return point


TOP_FACE_QUANTILE = .85
TOP_FACE_BAND_M = .003
# A block is 20-30 mm tall, so a top face legitimately sits at most that far
# above the mixed-surface centre. A larger jump means the box caught something
# else above the block (frame, fingers, a second red object) and is not trusted.
TOP_FACE_MAX_RISE_M = .045
# Below this the "top band" is just the mixed centre again, so the pick is a
# no-op: the reference may only ever be corrected upwards.
TOP_FACE_MIN_RISE_M = .0005


def top_face_camera_point(point, T_base_camera, *, min_points=12,
                          min_rise_m=TOP_FACE_MIN_RISE_M, max_rise_m=TOP_FACE_MAX_RISE_M):
    """Re-point a detection at the block's top face instead of the camera-facing one.

    ``locate_top_surface`` runs inside the vision worker, which has no robot
    pose, so it averages whatever the box covers: on 2026-09-18 the reference
    sat 2-6 mm below the real top face and slid by up to 25 mm between poses
    (2026-09-17 stationary-block audit). The worker therefore also ships a
    compact sample of the ROI point cloud (``surface_samples_xyz_m``); with the
    camera pose known here, the sample's upper Z band is the top face. The
    reference keeps the same camera-frame interface, so callers are unchanged.
    Detections without a sample (centre-pixel ``locate``) pass through as-is.
    """
    result = dict(point)
    samples = np.asarray(point.get('surface_samples_xyz_m') or [], float)
    if (not point.get('valid') or samples.ndim != 2 or samples.shape[0] < min_points
            or samples.shape[1] != 3):
        return result
    samples = samples[np.isfinite(samples).all(axis=1)]
    if len(samples) < min_points:
        return result
    T = np.asarray(T_base_camera, float)
    base = (T[:3, :3] @ samples.T).T + T[:3, 3]
    z = base[:, 2]
    band = base[np.abs(z-np.quantile(z, TOP_FACE_QUANTILE)) <= TOP_FACE_BAND_M]
    if len(band) < min_points:
        return result
    reference = band.mean(axis=0)
    centre = np.asarray(point['xyz_camera_m'], float)
    result['centre_reference_camera_xyz_m'] = centre.tolist()
    result['centre_reference_base_z_m'] = float((T @ np.r_[centre, 1.])[2])
    rise = float(reference[2])-result['centre_reference_base_z_m']
    if not min_rise_m <= rise <= max_rise_m:
        return result
    result['xyz_camera_m'] = np.linalg.solve(T[:3, :3], reference-T[:3, 3]).tolist()
    result['height_reference'] = 'top_face'
    result['reference_point_count'] = int(len(band))
    result['reference_base_z_m'] = float(reference[2])
    return result


def observe_at_fixed(controller, fd, preview, read_state, names, locate, event,
                     hold_tolerance_deg=HOLD_TOLERANCE_DEG, *, continuous_inference=False,
                     home=None):
    """Only detection/depth rejection is recoverable; robot faults propagate.

    Returns None on an explicit quit, otherwise a fresh sample, state, flange,
    and valid point. No actuator commands are sent by this function.

    ``hold_tolerance_deg`` is the parked-arm allowance: the arm can snap
    0.3-0.4 deg about 12-16 s after arriving, which must not stop the session.
    """
    anchor, _, _ = read_state()
    strikes = 0
    def monitor():
        nonlocal strikes
        controller.tick()
        if controller.locked:
            raise RuntimeError('固定点识别监控已停止')
        state, flange, _ = read_state()
        excursion = _trusted_excursion(state.joints, anchor.joints, hold_tolerance_deg)
        if excursion > np.deg2rad(GUARD_HARD_MARGIN_DEG):
            raise RuntimeError(f'识别期间机械臂偏离固定点超过{hold_tolerance_deg:g}°+{GUARD_HARD_MARGIN_DEG:g}°硬界限')
        if excursion > np.deg2rad(GUARD_NOISE_FLOOR_DEG):
            strikes += 1
            if strikes >= GUARD_PERSIST_SAMPLES:
                raise RuntimeError(f'连续{strikes}次识别期间机械臂偏离固定点超过{hold_tolerance_deg:g}°')
        else:
            strikes = 0
        return state, flange
    def poll():
        poll_key(fd)
        monitor()
    attempt = 0
    while True:
        attempt += 1
        preview.stage = 'Detecting red block at fixed pose'
        preview.detail = ''
        requested = time.monotonic()
        while True:
            poll()
            preview.pump()
            sample = preview.sample()
            if sample is not None and sample[0] >= requested:
                break
            time.sleep(.02)
        state, flange = monitor()
        point = target_point(sample, names, locate)
        if point['valid']:
            return sample, state, flange, point
        event(dict(event='observation_rejected', attempt=attempt, observation_stamp=sample[0], detail=point))
        # Stop competing with CAN while the operator adjusts the block.
        if not continuous_inference:
            preview.pause_inference(sample)
            preview.wait_until_paused(poll=poll)
        preview.stage = 'Detection rejected - terminal: retry / quit'
        preview.detail = str(point['reason'])
        home_text = 'home 回初始位置；' if home is not None else ''
        print(f'识别未通过：{point}。机械臂保持固定点，未发送接近目标。\n'
              f'调整物块后输入 retry 回车重新识别；{home_text}quit 退出并保持位置；'
              '空格/Esc/Ctrl+C 急停。', flush=True)
        def poll():
            monitor()
            preview.pump()

        while True:
            kind, typed = type_word(fd, poll)
            if kind == 'stop':
                raise KeyboardInterrupt('重新识别等待期间用户急停')
            if kind == 'eof':
                event(dict(event='observation_exit', reason='terminal_eof'))
                return None
            if typed == 'quit':
                event(dict(event='observation_exit', reason='quit'))
                return None
            if typed == 'retry':
                if not continuous_inference:
                    preview.resume_inference()
                event(dict(event='observation_retry', attempt=attempt+1))
                break
            if typed == 'home' and home is not None:
                event(dict(event='observation_home_requested'))
                home()
                event(dict(event='observation_home_reached'))
                return None
            words = 'retry / home / quit' if home is not None else 'retry / quit'
            print(f'请输入 {words}（Backspace/Delete 可改，Ctrl+U 清空）；'
                  '尚未获得有效目标，go 不会执行运动。', flush=True)


def wait_for_plan_retry(controller, fd, preview, read_state, anchor, event,
                        hold_tolerance_deg=HOLD_TOLERANCE_DEG, *, continuous_inference=False,
                        home=None):
    """Wait at the fixed pose after offline planning rejection; never move."""
    preview.stage = 'Plan rejected - terminal: retry / quit'
    home_text = 'home 回初始位置；' if home is not None else ''
    print('规划未通过，机械臂保持固定点。调整物块后输入 retry 重新识别；'
          f'{home_text}quit 退出并保持位置；空格/Esc/Ctrl+C 急停。', flush=True)
    strikes = 0

    def poll():
        nonlocal strikes
        controller.tick()
        if controller.locked:
            raise RuntimeError('规划重试监控已停止')
        state, _, _ = read_state()
        excursion = _trusted_excursion(state.joints, anchor.joints, hold_tolerance_deg)
        if excursion > np.deg2rad(GUARD_HARD_MARGIN_DEG):
            raise RuntimeError(f'规划或等待重试期间机械臂偏离固定点超过{hold_tolerance_deg:g}°+{GUARD_HARD_MARGIN_DEG:g}°硬界限')
        if excursion > np.deg2rad(GUARD_NOISE_FLOOR_DEG):
            strikes += 1
            if strikes >= GUARD_PERSIST_SAMPLES:
                raise RuntimeError(f'连续{strikes}次规划或等待重试期间机械臂偏离固定点超过{hold_tolerance_deg:g}°')
        else:
            strikes = 0
        preview.pump()

    while True:
        kind, typed = type_word(fd, poll)
        if kind == 'stop':
            raise KeyboardInterrupt('规划重试期间用户急停')
        if kind == 'eof':
            event(dict(event='planning_exit', reason='terminal_eof'))
            return False
        if typed == 'quit':
            event(dict(event='planning_exit', reason='quit'))
            return False
        if typed == 'retry':
            if not continuous_inference:
                preview.resume_inference()
            event(dict(event='planning_retry'))
            return True
        if typed == 'home' and home is not None:
            event(dict(event='planning_home_requested'))
            home()
            event(dict(event='planning_home_reached'))
            return False
        words = 'retry / home / quit' if home is not None else 'retry / quit'
        print(f'请输入 {words}（Backspace/Delete 可改）；被拒绝的规划不能用 go 执行。', flush=True)
