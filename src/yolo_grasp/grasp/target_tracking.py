"""Timestamped base-frame target monitoring and stop/observe/replan approach."""
from collections import deque
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .hover_observation import target_point, top_face_camera_point
from .hover_motion import execute_path


class TargetReacquisitionTimeout(TimeoutError):
    """Target stayed missing/unstable after a controlled motion stop."""


class TrackingReplanLimit(RuntimeError):
    """The safely stopped approach exhausted its permitted replans."""


class PoseHistory:
    """CAN poses indexed by their receive timestamps, not by inference completion."""
    def __init__(self):
        self.rows = deque()

    def add(self, flange, stamps):
        stamps = np.asarray(stamps, float)
        if not np.isfinite(stamps).all() or np.ptp(stamps) > .05:
            raise ValueError('跟踪位姿分帧时间差超过50 ms')
        stamp = float(np.mean(stamps))
        if self.rows and stamp < self.rows[-1][0]:
            raise ValueError('机器人时间戳倒退')
        if not self.rows or stamp > self.rows[-1][0]:
            self.rows.append((stamp, np.asarray(flange, float).copy()))
        while self.rows and self.rows[0][0] < stamp-5:
            self.rows.popleft()

    def at(self, stamp):
        for (a, first), (b, second) in zip(self.rows, list(self.rows)[1:]):
            if a <= stamp <= b:
                if b-a > .1:
                    raise ValueError('拍摄时刻机器人位姿间隔超过100 ms')
                fraction = (stamp-a)/(b-a)
                pose = np.eye(4)
                pose[:3, 3] = first[:3, 3]*(1-fraction)+second[:3, 3]*fraction
                pose[:3, :3] = Slerp([a, b], Rotation.from_matrix(
                    np.stack([first[:3, :3], second[:3, :3]])))([stamp]).as_matrix()[0]
                return pose
        raise ValueError('没有包围拍摄时刻的机器人位姿，拒绝外推')


class TargetTracker:
    def __init__(self, preview, history, handeye, names, locate, *, threshold_m=.010,
                 max_age=1., count_persist_frames=3, now=time.time):
        if not np.isfinite(threshold_m) or not .005 <= threshold_m <= .05:
            raise ValueError('移动阈值必须为5–50 mm')
        self.preview, self.history, self.handeye = preview, history, np.asarray(handeye)
        self.names, self.locate = names, locate
        self.threshold_m, self.max_age, self.now = threshold_m, max_age, now
        if not isinstance(count_persist_frames, int) or not 1 <= count_persist_frames <= 10:
            raise ValueError('目标数量异常确认帧数必须为1–10')
        self.count_persist_frames = count_persist_frames
        self.last_frame = None
        self.target = None
        self.strikes = 0
        self.count_strikes = 0
        self.count_history = deque(maxlen=count_persist_frames)

    def set_target(self, base):
        self.target = np.asarray(base, float).copy()
        self.strikes = 0
        self.count_strikes = 0
        self.count_history.clear()

    def observation(self):
        sample = self.preview.sample()
        if sample is None:
            return None, '等待新检测帧'
        _, frame, _, _ = sample
        metadata = frame.metadata
        domain = metadata.get('timestamp_domain', '')
        if domain not in ('timestamp_domain.global_time', 'timestamp_domain.system_time'):
            return None, '相机时间戳未同步至主机，不能进行运动中定位'
        stamp = metadata.get('color_timestamp_ms', float('nan'))/1000
        age = self.now()-stamp
        if not np.isfinite(age) or age < -.02 or age > self.max_age:
            return None, '视觉观测过期或时钟不同步'
        if self.last_frame is not None and stamp <= self.last_frame:
            return None, None
        self.last_frame = stamp
        try:
            flange = self.history.at(stamp)
        except ValueError as error:
            return None, str(error)
        point = target_point(sample, self.names, self.locate)
        if not point['valid']:
            reason = str(point.get('reason', '目标深度无效'))
            if reason == 'red_block_count':
                return None, f'red_block_count(count={point.get("count", "unknown")})'
            return None, reason
        point = top_face_camera_point(point, flange @ self.handeye)
        camera = np.asarray(point['xyz_camera_m'], float)
        if camera.shape != (3,) or not np.isfinite(camera).all():
            return None, '目标三维坐标无效'
        base = (flange @ self.handeye @ np.r_[camera, 1])[:3]
        return dict(stamp=stamp, base=base, sample=sample, point=point), None

    def monitor(self):
        observation, reason = self.observation()
        if reason:
            self.strikes = 0
            if reason.startswith('red_block_count('):
                self.count_strikes += 1
                self.count_history.append(reason)
                if self.count_strikes < self.count_persist_frames:
                    return None
                counts = ','.join(item.partition('count=')[2].rstrip(')')
                                  for item in self.count_history)
                return (f'red_block_count连续{self.count_strikes}帧'
                        f'（各帧数量：{counts}）')
            self.count_strikes = 0
            self.count_history.clear()
            return reason
        if observation is None:
            return None
        self.count_strikes = 0
        self.count_history.clear()
        distance = float(np.linalg.norm(observation['base']-self.target))
        self.strikes = self.strikes+1 if distance > self.threshold_m else 0
        if self.strikes >= 2:
            return f'红块基座坐标连续两帧变化{distance*1000:.1f} mm'
        return None


def wait_stable_target(tracker, tick, event, *, timeout=15., now=time.monotonic,
                       wall=time.time, sleep=time.sleep):
    """Only post-stop frames count; missing/ambiguous targets never resume motion."""
    requested, deadline = wall(), now()+timeout
    observations = []
    reason = '等待目标'
    while now() < deadline:
        tick()
        observation, invalid = tracker.observation()
        if invalid:
            observations.clear()
            reason = invalid
        elif observation is not None and observation['stamp'] >= requested:
            if observations and np.linalg.norm(observation['base']-observations[0]['base']) > .005:
                observations.clear()
            observations.append(observation)
            if len(observations) >= 3 and observation['stamp']-observations[0]['stamp'] >= .3:
                event(dict(event='tracking_target_stable', base_xyz_m=observation['base'].tolist(),
                           camera_stamp_s=observation['stamp']))
                return observation
        sleep(.01)
    raise TargetReacquisitionTimeout(f'停稳后{timeout:g}秒内未获得稳定红块：{reason}')


def tracked_approach(controller, tracker, report, trajectory, build_plan, tick, event,
                     *, hold_tolerance_deg=.8, max_replans=5, monitor=None,
                     verify_after_reached=True):
    """Return only after arrival AND a fresh stationary target check before closure."""
    replans = 0
    while True:
        tracker.preview.stage = 'Checking red block before motion / gripper close'
        observation = wait_stable_target(tracker, tick, event)
        target = np.asarray(report['red_block_base_xyz_m'])
        tracker.preview.detail = 'Target base XYZ mm: ' + ', '.join(f'{v*1000:.1f}' for v in target)
        changed = np.linalg.norm(observation['base']-target) > tracker.threshold_m
        if changed:
            if replans >= max_replans:
                raise TrackingReplanLimit('红块重规划次数超过上限，停止本次抓取')
            replans += 1
            print(f'红块目标已变化，正在从当前位置重规划（{replans}/{max_replans}）。', flush=True)
            tracker.preview.stage = 'Replanning from stopped current pose'
            report, trajectory = build_plan(observation)
            event(dict(event='tracking_replan', attempt=replans, diagnostic=report))
            # The target can move during planning: check fresh images before executing.
            continue
        tracker.set_target(target)
        tracker.preview.stage = 'Tracking red block during approach'
        reached = execute_path(controller, *trajectory, tick, event,
                               hold_tolerance_deg=hold_tolerance_deg,
                               monitor=tracker.monitor if monitor is None else monitor)
        if reached and not verify_after_reached:
            event(dict(event='tracking_pregrasp_locked', base_xyz_m=target.tolist()))
            return report
        tracker.preview.stage = 'Stopped - reacquiring red block'
        observation = wait_stable_target(tracker, tick, event)
        if reached and np.linalg.norm(observation['base']-target) <= tracker.threshold_m:
            event(dict(event='tracking_pregrasp_verified', base_xyz_m=observation['base'].tolist()))
            return report
        if replans >= max_replans:
            raise TrackingReplanLimit('红块重规划次数超过上限，停止本次抓取')
        replans += 1
        print(f'已停稳并重新定位，正在从当前位置重规划（{replans}/{max_replans}）。', flush=True)
        tracker.preview.stage = 'Replanning from stopped current pose'
        report, trajectory = build_plan(observation)
        event(dict(event='tracking_replan', attempt=replans, diagnostic=report))
