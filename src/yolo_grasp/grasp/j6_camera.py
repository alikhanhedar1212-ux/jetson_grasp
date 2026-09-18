"""Stationary D405 observations for J6 diagnostics; no robot commands."""
import hashlib
import json
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from .camera import RealSense
from .geometry import transform, pose_error
from .handeye import BOARD, SERIAL, detect_board, estimate_board
from .handeye_capture import PoseHistory


def ordered_corners(corners, previous):
    """Track the first image's arbitrary origin through small J6 increments."""
    if previous is None:
        return corners, False
    scores = [float(np.sqrt(np.mean(np.sum((c - previous) ** 2, axis=1))))
              for c in (corners, corners[::-1])]
    reverse = scores[1] < scores[0]
    if min(scores) > 60 or abs(scores[0] - scores[1]) < 10:
        raise ValueError('棋盘角点顺序不确定或位移过大；拒绝用于一致性分析')
    return (corners[::-1].copy() if reverse else corners), reverse


class Observation:
    def __init__(self, output, calibration):
        self.output = Path(output)
        raw = Path(calibration).read_bytes()
        self.calibration = json.loads(raw)
        if self.calibration['camera_serial'] != SERIAL or self.calibration['board'] != BOARD:
            raise ValueError('手眼文件相机或棋盘规格不匹配')
        # Existing handeye solver names its color-camera frame "wrist".
        self.x = transform(self.calibration['T_flange_wrist'])
        self.info = dict(camera_serial=SERIAL, board=BOARD,
                         handeye_path=str(Path(calibration).resolve()),
                         handeye_sha256=hashlib.sha256(raw).hexdigest(),
                         handeye_validated=self.calibration.get('validated', False),
                         T_flange_camera=self.x.tolist(),
                         sync='stationary CAN bracket; not hardware synchronized',
                         origin='first detected corner; temporal order tracking, not physical origin confirmation')
        self.camera = self.history = None
        self.previous = None
        self.rows = []

    def start(self, receiver):
        self.history = PoseHistory(receiver)
        cfg = dict(serial=SERIAL, color=dict(width=640, height=480, fps=30),
                   depth=dict(width=640, height=480, fps=30), warmup_frames=30,
                   timeout_ms=1000, max_rgb_depth_skew_ms=35)
        self.camera = RealSense(cfg)
        self.camera.__enter__()

    def close(self):
        try:
            if self.camera is not None:
                self.camera.__exit__()
        finally:
            if self.history is not None:
                self.history.close()

    def collect(self, sample, target, step):
        # The main thread continues keyboard and robot monitoring during USB,
        # OpenCV and disk operations. Worker never issues robot commands.
        result = queue.Queue()
        def work():
            try:
                result.put((True, self._collect(sample, target)))
            except Exception as error:
                result.put((False, error))
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        deadline = time.monotonic() + 20
        while worker.is_alive():
            step()
            if time.monotonic() > deadline:
                raise TimeoutError('D405 采集处理超时')
            worker.join(.02)
        step()
        ok, value = result.get_nowait()
        if not ok:
            raise value
        self.rows.append(value)
        return value

    def _collect(self, sample, target):
        start = time.time()
        frame = self.camera.capture(discard=5)
        end = time.time()
        time.sleep(.25)
        robot, bracket = self.history.window(start - .3, time.time())
        for r in bracket:
            if (not all(r['enabled']) or r['arrival_flag'] != 0 or
                    np.max(abs(np.array(r['joints_rad']) - target)) > np.deg2rad(.2)):
                raise RuntimeError('拍摄期间未保持目标姿态/使能/停止状态')
        frame.metadata.update(capture_host_start_s=start, capture_host_end_s=end)
        directory = self.output / f"sample_{sample['index']:04d}"
        frame.save(directory)
        value = dict(index=sample['index'], offset_deg=sample['offset_deg'],
                     valid=False, T_base_flange=robot['T_base_flange'],
                     robot_bracket=bracket, capture_host_start_s=start, capture_host_end_s=end,
                     color_intrinsics=frame.metadata['color_intrinsics'],
                     origin_operator_confirmed=False)
        try:
            corners, reverse = ordered_corners(detect_board(frame.bgr), self.previous)
            board, rms, rays = estimate_board(corners, frame.metadata['color_intrinsics'])
            value.update(valid=True, corners_px=corners.tolist(), normalized_corners=rays.tolist(),
                         corner_order_reversed=reverse, pnp_rms_px=rms,
                         T_camera_board=board.tolist(),
                         T_base_board=(np.array(robot['T_base_flange']) @ self.x @ board).tolist())
            self.previous = corners.copy()
            view = frame.bgr.copy()
            cv2.drawChessboardCorners(view, (9, 6), corners.astype(np.float32).reshape(-1, 1, 2), True)
            for i, label in ((0, '0'), (8, '+X'), (45, '+Y')):
                cv2.putText(view, label, tuple(corners[i].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 2)
            if not cv2.imwrite(str(directory / 'corners.png'), view):
                raise OSError('角点图保存失败')
        except ValueError as error:
            value['error'] = str(error)
        (directory / 'observation.json').write_text(json.dumps(value, indent=2) + '\n')
        return value

    def report(self):
        valid = [r for r in self.rows if r['valid']]
        lines = ['# J6 + D405 固定棋盘诊断', '',
                 f'有效观测 {len(valid)}/{len(self.rows)}；手眼候选 validated={self.info["handeye_validated"]}。',
                 '未重新拟合手眼；以下相对第一个有效观测，不能作为标定验收。', '',
                 '| 点 | J6 偏移 ° | PnP RMS px | 棋盘基座位移 mm | 姿态变化 ° |',
                 '|---|---:|---:|---:|---:|']
        for r in valid:
            d, a = pose_error(r['T_base_board'], valid[0]['T_base_board'])
            lines.append(f"| {r['index']} | {r['offset_deg']:g} | {r['pnp_rms_px']:.4f} | {d*1000:.4f} | {np.rad2deg(a):.4f} |")
        lines += ['', '无效点：'] + [f"- {r['index']}: {r['error']}" for r in self.rows if not r['valid']]
        lines += ['', '需查看 corners.png 确认每点角点原点相同；当前采用图像连续性跟踪，不替代物理标记。',
                  '只有单轴旋转的数据不能独立求解完整手眼外参。同步采用停稳时间窗，未校验曝光时刻与主机时钟的严格对应。', '']
        (self.output / 'BOARD_REPORT.md').write_text('\n'.join(lines))
