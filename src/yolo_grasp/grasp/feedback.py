"""Arrival checks independent of SDK cached-message ownership."""
from dataclasses import dataclass
import numpy as np
from .geometry import pose_error


@dataclass
class Feedback:
    tcp: np.ndarray
    component_timestamps: tuple
    status_timestamp: float
    motion_status: int
    arm_status: int
    error_status: int


class Arrival:
    def __init__(self, target, issued_wall_s, cfg):
        self.target, self.issued, self.cfg = target, issued_wall_s, cfg
        self.since = None

    def update(self, sample, wall_s, monotonic_s):
        stamps = np.asarray((*sample.component_timestamps, sample.status_timestamp), dtype=float)
        if len(stamps) != 4 or not np.isfinite(stamps).all():
            raise RuntimeError("Missing complete timestamped pose/status feedback")
        age = wall_s - stamps
        if np.any(age < 0) or np.any(age > self.cfg["max_feedback_age_s"]):
            raise RuntimeError("Stale feedback or mismatched clock domain")
        if np.ptp(stamps[:3]) > self.cfg["max_pose_skew_s"]:
            raise RuntimeError("Pose CAN components are too far apart in time")
        if sample.arm_status != 0 or sample.error_status != 0:
            raise RuntimeError(f"Arm fault: {sample.arm_status}/{sample.error_status}")
        pos, angle = pose_error(sample.tcp, self.target)
        arrived = (np.all(stamps > self.issued) and sample.motion_status == 0
                   and pos <= self.cfg["position_tolerance_m"]
                   and angle <= self.cfg["angle_tolerance_rad"])
        if not arrived:
            self.since = None
            return False
        if self.since is None:
            self.since = monotonic_s
        return monotonic_s - self.since >= self.cfg["settle_s"]
