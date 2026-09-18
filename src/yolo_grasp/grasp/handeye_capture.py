"""Read-only CAN + D405 manual hand-eye capture. Never controls the robot."""
from collections import deque
import json
from pathlib import Path
import struct
import threading
import time

import cv2
import numpy as np
from .arm_console import JointReceiver, decode_state
from .camera import RealSense
from .geometry import pose_matrix, pose_error
from .handeye import BOARD, SERIAL, detect_board, estimate_board

POSE_IDS = (0x2A2, 0x2A3, 0x2A4)


class PoseReceiver(JointReceiver):
    IDS = JointReceiver.IDS + POSE_IDS


def decode_pose(frames, now):
    state = decode_state(frames, now)
    state.healthy()
    if any(k not in frames for k in POSE_IDS):
        raise ValueError('Incomplete flange feedback')
    stamps = np.array([frames[k][1] for k in POSE_IDS])
    if not np.isfinite(stamps).all() or np.any(now - stamps < 0) or np.any(now - stamps > .15) or np.ptp(stamps) > .03:
        raise ValueError('Stale/skewed flange fragments')
    if any(len(frames[k][0]) != 8 for k in POSE_IDS):
        raise ValueError('Invalid flange CAN payload')
    values = np.array([v for k in POSE_IDS for v in struct.unpack('>ii', frames[k][0])], float)
    values[:3] *= 1e-6
    values[3:] *= np.pi / 180000
    # Matches the inspected SDK get_flange_pose: Rz(yaw) Ry(pitch) Rx(roll).
    return {'host_s': now, 'T_base_flange': pose_matrix(values).tolist(),
            'flange_pose_m_rad': values.tolist(), 'pose_stamps': stamps.tolist(),
            'joints_rad': state.joints.tolist(), 'enabled': list(state.enabled),
            'arm_status': state.arm_status, 'error_status': state.error_status,
            'driver_faults': list(state.driver_faults), 'arrival_flag': state.motion_status}


class PoseHistory:
    def __init__(self, receiver):
        self.receiver = receiver
        self.samples = deque(maxlen=3000)
        self.lock = threading.Lock()
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        while not self.cancel.is_set():
            now = time.time()
            try:
                sample = decode_pose(self.receiver.snapshot(), now)
            except Exception as error:
                sample = {'host_s': now, 'error': str(error)}
            with self.lock:
                self.samples.append(sample)
            self.cancel.wait(.01)

    def close(self):
        self.cancel.set()
        self.thread.join(timeout=1)

    def window(self, start, end):
        with self.lock:
            samples = [s for s in self.samples if start <= s['host_s'] <= end]
        return stable_window(samples, start, end)


def stable_window(samples, start, end):
    if len(samples) < 10 or samples[0]['host_s'] - start > .05 or end - samples[-1]['host_s'] > .05:
        raise ValueError('Robot history does not bracket image capture; wait and retry')
    if any('error' in s for s in samples):
        raise ValueError('Robot fault or incomplete/stale CAN during image capture')
    if np.max(np.diff([s['host_s'] for s in samples])) > .05:
        raise ValueError('Gap in robot feedback history')
    stamps = [min(s['pose_stamps']) for s in samples]
    if stamps[-1] - stamps[0] < end - start - .2:
        raise ValueError('Robot feedback did not advance through capture interval')
    if any(s.get('arrival_flag', 0) != 0 for s in samples):
        raise ValueError('Robot motion is not complete; wait before saving')
    anchor = np.array(samples[0]['T_base_flange'])
    joints = np.array(samples[0]['joints_rad'])
    for s in samples:
        p, r = pose_error(anchor, s['T_base_flange'])
        if p > .001 or r > np.deg2rad(.2) or np.max(abs(np.array(s['joints_rad']) - joints)) > np.deg2rad(.2):
            raise ValueError('Robot moved during capture (>1 mm or 0.2 deg); sample rejected')
    return samples[len(samples) // 2], samples


def prepare_session(directory, manifest, resume=False):
    """Create a capture session or safely continue it without overwriting samples."""
    directory = Path(directory)
    if not resume:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'session.json').write_text(json.dumps(manifest, indent=2))
        return 0, 'train', []
    if not directory.is_dir() or not (directory / 'session.json').is_file():
        raise ValueError('--resume 需要已有且包含 session.json 的采集目录')
    existing = json.loads((directory / 'session.json').read_text())
    for key in ('schema_version', 'camera_serial', 'board', 'robot_frame', 'robot_model', 'channel'):
        if existing.get(key) != manifest.get(key):
            raise ValueError(f'不能续采：session.json 的 {key} 与当前配置不一致')
    if existing.get('firmware', {}).get('software_version') != manifest['firmware'].get('software_version'):
        raise ValueError('不能续采：机械臂固件版本与原会话不一致')
    indices, saved, split = [], [], 'train'
    for sample_dir in sorted(directory.glob('sample_*')):
        suffix = sample_dir.name.removeprefix('sample_')
        if suffix.isdigit():
            indices.append(int(suffix))
        record_path = sample_dir / 'sample.json'
        if not record_path.is_file():
            continue  # Preserve an interrupted partial folder; never reuse its number.
        record = json.loads(record_path.read_text())
        if record.get('camera_serial') != SERIAL or record.get('split') not in ('train', 'validation'):
            raise ValueError(f'不能续采：样本身份无效：{record_path}')
        saved.append(np.asarray(record['T_base_flange'], float))
        split = record['split']
    count = max(indices, default=-1)+1
    return count, split, saved


def capture(args):
    # Query only the verified SDK firmware API; disconnect SDK before capture.
    from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile
    arm = AgxArmFactory.create_arm(create_agx_arm_config(robot='piper_x', firmeware_version='default',
        interface='socketcan', channel=args.channel))
    try:
        arm.connect()
        from .j6_experiment import query_firmware
        firmware = query_firmware(arm)
        profile = resolve_firmware_profile('piper_x', firmware['software_version'])
    finally:
        arm.disconnect()
    manifest = {'schema_version': 1, 'camera_serial': SERIAL, 'board': BOARD,
        'robot_frame': 'flange', 'robot_model': 'piper_x', 'channel': args.channel,
        'firmware': firmware, 'sdk_profile': profile, 'created_host_s': time.time(),
        'pose_convention': 'metres; R=Rz(yaw)Ry(pitch)Rx(roll); T_A_B maps B to A',
        'sync': 'stationary bracket, not hardware trigger or cross-device timestamp equality',
        'board_origin': 'operator confirms same physical corner 0 and +X corner 8 in every image'}
    count, split, saved = prepare_session(args.session, manifest, getattr(args, 'resume', False))
    receiver = PoseReceiver(args.channel)
    history = PoseHistory(receiver)
    cfg = {'serial': SERIAL, 'color': {'width': 640, 'height': 480, 'fps': 30},
           'depth': {'width': 640, 'height': 480, 'fps': 30}, 'warmup_frames': 30,
           'timeout_ms': 5000, 'max_rgb_depth_skew_ms': 35}
    flip = False
    try:
        with RealSense(cfg) as camera:
            if getattr(args, 'resume', False):
                print(f'续采模式：从 sample_{count:04d} 开始；当前类别={split}；'
                      f'已加载 {len(saved)} 个完整样本。')
            print('S=save (confirms physical corner 0); F=reverse corner order; V=train/validation; Q/Esc=exit.')
            print('READ ONLY: camera keys do NOT stop or control the arm. Use its separate control terminal/hardware stop.')
            while True:
                start = time.time()
                frame = camera.capture(discard=2)
                end = time.time()
                frame.metadata.update(capture_host_start_s=start, capture_host_end_s=end)
                view = frame.bgr.copy()
                try:
                    corners = detect_board(frame.bgr, flip)
                    board, rms, rays = estimate_board(corners, frame.metadata['color_intrinsics'])
                    cv2.drawChessboardCorners(view, (9, 6), corners.astype(np.float32).reshape(-1, 1, 2), True)
                    for i, name in ((0, '0 ORIGIN'), (8, '+X'), (45, '+Y')):
                        cv2.putText(view, name, tuple(corners[i].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2)
                    info = f'{split} | {count} saved | RMS {rms:.3f}px | F flip={flip}'
                except ValueError as error:
                    corners = None
                    info = str(error)
                cv2.putText(view, info[:90], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 255), 1)
                cv2.imshow('D405 hand-eye: read-only', view)
                key = cv2.waitKey(1) & 0xff
                if key in (27, ord('q')):
                    break
                if key == ord('f'):
                    flip = not flip
                if key == ord('v'):
                    split = 'validation' if split == 'train' else 'train'
                if key != ord('s'):
                    continue
                try:
                    if corners is None:
                        raise ValueError('No valid full board')
                    time.sleep(.25)
                    sample, bracket = history.window(start - .5, time.time())
                    t = np.array(sample['T_base_flange'])
                    if any(pose_error(t, old)[0] < .003 and pose_error(t, old)[1] < np.deg2rad(2) for old in saved):
                        raise ValueError('Near-duplicate robot pose; change viewpoint')
                    directory = args.session / f'sample_{count:04d}'
                    # A failed save is kept as an incomplete folder, never overwritten.
                    count += 1
                    frame.save(directory)
                    cv2.imwrite(str(directory / 'corners.png'), view)
                    record = {'schema_version': 1, 'camera_serial': SERIAL, 'split': split,
                        'T_base_flange': t.tolist(), 'T_camera_board': board.tolist(),
                        'corners_px': corners.tolist(), 'normalized_corners': rays.tolist(),
                        'color_intrinsics': frame.metadata['color_intrinsics'], 'pnp_rms_px': rms,
                        'corner_order_reversed': flip, 'origin_operator_confirmed': True,
                        'robot_bracket': bracket, 'capture_host_start_s': start, 'capture_host_end_s': end}
                    (directory / 'sample.json').write_text(json.dumps(record, indent=2))
                    saved.append(t)
                    print(f'Saved {directory.name} ({split}); total valid={len(saved)}')
                except (ValueError, RuntimeError, OSError) as error:
                    print(f'Sample rejected: {error}')
    finally:
        history.close()
        receiver.close()
        cv2.destroyAllWindows()
