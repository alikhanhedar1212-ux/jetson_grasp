"""Piper X adapter. Construction does not connect; methods never run at import.

A separate receive-only SocketCAN socket keeps immutable copies of each pose
fragment. The SDK's merged pose timestamp cannot prove freshness of all axes.
"""
import struct
import threading
import time
import logging
import numpy as np
from .feedback import Arrival, Feedback
from .geometry import matrix_pose, pose_matrix
from .planning import check_workspace
from .config import validate_motion


class IncompleteFeedback(RuntimeError):
    pass


class CanFeedback:
    IDS = (0x2A1, 0x2A2, 0x2A3, 0x2A4, 0x2A8)

    def __init__(self, channel):
        import can
        self.bus = can.Bus(interface="socketcan", channel=channel,
                           can_filters=[{"can_id": i, "can_mask": 0x7FF, "extended": False} for i in self.IDS])
        self.frames = {}
        self.receive_stats = dict(messages=0, last_batch=0, last_lag_ms=0., max_lag_ms=0.)
        self.lock = threading.Lock()
        self.error = None
        self.cancel = threading.Event()
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            while not self.cancel.is_set():
                msg = self.bus.recv(timeout=.05)
                updates, count = {}, 0
                deadline = time.monotonic()+.002
                # Drain queued frames in bounded batches, publishing only the newest
                # per ID. Keep the original kernel timestamps: stale input stays stale.
                while msg is not None:
                    if not msg.is_error_frame and not msg.is_remote_frame and msg.dlc == 8:
                        stamp = float(msg.timestamp)
                        prior = updates.get(msg.arbitration_id)
                        if prior is None or stamp >= prior[1]:
                            updates[msg.arbitration_id] = (bytes(msg.data), stamp)
                    count += 1
                    if count >= 256 or time.monotonic() >= deadline or self.cancel.is_set():
                        break
                    msg = self.bus.recv(timeout=0)
                if updates:
                    lag = max(0., time.time()-min(v[1] for v in updates.values()))*1000
                    with self.lock:
                        self.frames.update(updates)
                        self.receive_stats['messages'] += count
                        self.receive_stats['last_batch'] = count
                        self.receive_stats['last_lag_ms'] = lag
                        self.receive_stats['max_lag_ms'] = max(self.receive_stats['max_lag_ms'], lag)
        except Exception as error:
            self.error = error

    def snapshot(self):
        if self.error is not None:
            raise RuntimeError("CAN feedback listener failed") from self.error
        with self.lock:
            return dict(self.frames)

    def diagnostics(self):
        with self.lock:
            return dict(self.receive_stats)

    def close(self):
        self.cancel.set()
        self.thread.join(timeout=1)
        self.bus.shutdown()


def decode_feedback(frames, T_flange_tcp):
    if any(i not in frames for i in (0x2A1, 0x2A2, 0x2A3, 0x2A4)):
        raise IncompleteFeedback("Incomplete CAN pose/status feedback")
    values = np.array([v for i in (0x2A2, 0x2A3, 0x2A4)
                       for v in struct.unpack(">ii", frames[i][0])], dtype=float)
    values[:3] *= 1e-6
    values[3:] *= np.pi / 180000
    status, timestamp = frames[0x2A1]
    return Feedback(pose_matrix(values) @ T_flange_tcp,
                    tuple(frames[i][1] for i in (0x2A2, 0x2A3, 0x2A4)), timestamp,
                    status[4], status[1], int.from_bytes(status[6:8], "big"))


def inspect_piper(cfg):
    """Bootstrap firmware/pose inspection without cameras, calibration or enabling."""
    from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config, resolve_firmware_profile
    rc = cfg["robot"]
    # The default profile is used only to send the version-independent firmware
    # query. It is never selected implicitly for an actual motion session.
    profile = resolve_firmware_profile(ArmModel.PIPER_X, rc["firmware"]) if rc.get("firmware") else "default"
    arm = AgxArmFactory.create_arm(create_agx_arm_config(
        robot=ArmModel.PIPER_X, firmeware_version=profile,
        interface="socketcan", channel=rc["can_interface"]))
    receiver = None
    try:
        receiver = CanFeedback(rc["can_interface"])
        arm.connect()
        firmware = arm.get_firmware(timeout=3)
        if firmware is None:
            raise RuntimeError("Firmware query timed out")
        sample = decode_feedback(receiver.snapshot(), np.eye(4))
        return {"firmware": firmware, "flange_pose_base": matrix_pose(sample.tcp),
                "pose_component_timestamps_s": sample.component_timestamps,
                "status_timestamp_s": sample.status_timestamp,
                "motion_status": sample.motion_status, "arm_status": sample.arm_status,
                "error_status": sample.error_status, "host_wall_s": time.time()}
    finally:
        try:
            if receiver is not None:
                receiver.close()
        finally:
            arm.disconnect()


class Piper:
    def __init__(self, cfg, calib):
        self.cfg, self.calib = cfg, calib
        self.arm = self.receiver = None
        self.holding_width = None

    def __enter__(self):
        validate_motion(self.cfg)
        from pyAgxArm import AgxArmFactory, ArmModel, create_agx_arm_config, resolve_firmware_profile
        rc = self.cfg["robot"]
        profile = resolve_firmware_profile(ArmModel.PIPER_X, rc["firmware"])
        self.arm = AgxArmFactory.create_arm(create_agx_arm_config(
            robot=ArmModel.PIPER_X, firmeware_version=profile,
            interface="socketcan", channel=rc["can_interface"]))
        try:
            self.receiver = CanFeedback(rc["can_interface"])
            self.arm.connect()
            firmware = self.arm.get_firmware(timeout=3)
            if firmware is None or firmware["software_version"] != rc["firmware"]:
                raise RuntimeError("Configured firmware does not match device firmware")
            self.gripper = self.arm.init_effector(self.arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            # Deliberately do not enable the arm automatically.
            self.wait_stationary_tcp()
        except BaseException:
            self.__exit__()
            raise
        return self

    def sample(self):
        return decode_feedback(self.receiver.snapshot(), self.calib["T_flange_tcp"])

    def stationary_tcp(self):
        """Check one fresh, fault-free arrival sample; not a settling interval."""
        sample = self.sample()
        guard = Arrival(sample.tcp, 0, self.cfg["feedback"])
        guard.update(sample, time.time(), time.monotonic())
        if sample.motion_status != 0:
            raise RuntimeError("Arm has not reported arrival")
        return sample.tcp.copy()

    def wait_stationary_tcp(self):
        """Observe a complete stable interval before taking a camera snapshot."""
        guard = None
        deadline = time.monotonic() + self.cfg["feedback"]["motion_timeout_s"]
        while time.monotonic() < deadline:
            try:
                sample = self.sample()
            except IncompleteFeedback:
                guard = None
                time.sleep(.02)
                continue
            if guard is None:
                guard = Arrival(sample.tcp.copy(), 0, self.cfg["feedback"])
            if guard.update(sample, time.time(), time.monotonic()):
                return sample.tcp.copy()
            from .geometry import pose_error
            position, angle = pose_error(sample.tcp, guard.target)
            if (position > self.cfg["feedback"]["position_tolerance_m"]
                    or angle > self.cfg["feedback"]["angle_tolerance_rad"]):
                guard = Arrival(sample.tcp.copy(), 0, self.cfg["feedback"])
            time.sleep(.02)
        raise TimeoutError("Arm did not remain stationary for the configured interval")

    def _motion_ready(self):
        if self.cfg["robot"].get("motion_enabled") is not True:
            raise RuntimeError("Motion is disabled in local config")
        # This flag records physical trajectory verification, not SDK capability.
        if self.cfg["robot"].get("paths_verified") is not True:
            raise RuntimeError("Transfer paths have not been verified")
        if self.arm.get_joints_enable_status_list() != [True] * 6:
            raise RuntimeError("All six joints must already be enabled")
        self.stationary_tcp()

    def move(self, target, linear):
        self._motion_ready()
        check_workspace(target, self.cfg["grasp"])
        flange = target @ np.linalg.inv(self.calib["T_flange_tcp"])
        check_workspace(flange, self.cfg["grasp"])
        self.arm.set_speed_percent(self.cfg["robot"]["speed_percent"])
        issued = time.time()
        guard = Arrival(target, issued, self.cfg["feedback"])
        deadline = time.monotonic() + self.cfg["feedback"]["motion_timeout_s"]
        try:
            (self.arm.move_l if linear else self.arm.move_p)(matrix_pose(flange))
            while time.monotonic() < deadline:
                if self.holding_width is not None:
                    self.check_hold()
                if guard.update(self.sample(), time.time(), time.monotonic()):
                    return
                time.sleep(.02)
            raise TimeoutError("Motion did not reach and settle at target")
        except BaseException:
            try:
                self.stop()
            except Exception:
                logging.exception("Could not send stop after motion failure")
            raise

    def _grip(self, width, expected, closing):
        self._motion_ready()
        issued = time.time()
        self.gripper.move_gripper_m(value=width, force=self.cfg["robot"]["gripper_force_n"])
        deadline = time.monotonic() + self.cfg["feedback"]["motion_timeout_s"]
        since = None
        while time.monotonic() < deadline:
            frame = self.receiver.snapshot().get(0x2A8)
            self.stationary_tcp()
            if frame is not None:
                data, stamp = frame
                if time.time() - stamp > self.cfg["feedback"]["max_feedback_age_s"]:
                    raise RuntimeError("Stale gripper feedback")
                # Width-mode frame, per local SDK codec: int32 micrometres,
                # int16 millinewtons, FOC bits, control mode.
                value = int.from_bytes(data[:4], "big", signed=True) * 1e-6
                force = int.from_bytes(data[4:6], "big", signed=True) * 1e-3
                if data[7] != 0 or stamp > time.time():
                    raise RuntimeError("Expected width-mode gripper feedback in host clock domain")
                if data[6] & 0x3F:
                    raise RuntimeError("Gripper driver reports a fault")
                ok = (stamp > issued and abs(value - expected) <= self.cfg["robot"]["gripper_tolerance_m"]
                      and (not closing or force >= self.cfg["robot"]["min_hold_force_n"]))
                if ok:
                    since = time.monotonic() if since is None else since
                    if time.monotonic() - since >= self.cfg["feedback"]["settle_s"]:
                        return
                else:
                    since = None
            else:
                since = None
            time.sleep(.02)
        raise TimeoutError("Gripper opening/contact was not confirmed")

    def open(self, width):
        self._grip(width, width, False)
        self.holding_width = None

    def close(self, expected_width):
        self._grip(0., expected_width, True)
        self.holding_width = expected_width

    def check_hold(self):
        frame = self.receiver.snapshot().get(0x2A8)
        if frame is None:
            raise RuntimeError("Missing gripper feedback while holding")
        data, stamp = frame
        age = time.time() - stamp
        value = int.from_bytes(data[:4], "big", signed=True) * 1e-6
        force = int.from_bytes(data[4:6], "big", signed=True) * 1e-3
        if (not 0 <= age <= self.cfg["feedback"]["max_feedback_age_s"]
                or data[7] != 0 or data[6] & 0x3F
                or abs(value - self.holding_width) > self.cfg["robot"]["gripper_tolerance_m"]
                or force < self.cfg["robot"]["min_hold_force_n"]):
            raise RuntimeError("Gripper lost verified contact while moving")

    def stop(self):
        if self.arm is not None:
            self.arm.electronic_emergency_stop()

    def __exit__(self, *args):
        try:
            if self.receiver is not None:
                self.receiver.close()
        finally:
            if self.arm is not None:
                self.arm.disconnect()
