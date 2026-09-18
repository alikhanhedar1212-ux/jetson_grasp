"""Interactive Piper X joint homing. No camera, gripper or encoder calibration."""
import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import select
import signal
import struct
import sys
import termios
import time
import tty

import numpy as np

from .robot import CanFeedback


JOINT_IDS = (0x2A5, 0x2A6, 0x2A7)
DRIVER_IDS = tuple(range(0x261, 0x267))


class JointReceiver(CanFeedback):
    IDS = (0x2A1,) + JOINT_IDS + DRIVER_IDS


@dataclass
class JointState:
    joints: np.ndarray
    stamps: np.ndarray
    enabled: tuple
    arm_status: int
    error_status: int
    driver_faults: tuple
    motion_status: int
    ctrl_mode: int = 0
    mode_feedback: int = 0

    def healthy(self):
        if self.arm_status or self.error_status or any(self.driver_faults):
            raise RuntimeError(f"机械臂故障: status={self.arm_status}, "
                               f"error={self.error_status}, drivers={self.driver_faults}")


def decode_state(frames, now, max_age=.25):
    ids = JointReceiver.IDS
    if any(i not in frames for i in ids):
        raise RuntimeError("等待完整的关节、机械臂状态和六个驱动器反馈")
    stamps = np.array([frames[i][1] for i in ids], dtype=float)
    if (not np.isfinite(stamps).all() or np.any(now - stamps < 0)
            or np.any(now - stamps > max_age)):
        raise RuntimeError("CAN 反馈过期或时钟不一致")
    if np.ptp(stamps[1:4]) > .05:
        raise RuntimeError("三个关节反馈分帧时间差过大")
    if any(len(frames[i][0]) != 8 for i in ids):
        raise RuntimeError("CAN 反馈长度错误")
    joints = np.array([v for i in JOINT_IDS for v in struct.unpack(">ii", frames[i][0])])
    status = frames[0x2A1][0]
    flags = [frames[i][0][5] for i in DRIVER_IDS]
    return JointState(joints * np.pi / 180000, stamps,
                      tuple(bool(f & 0x40) for f in flags), status[1],
                      int.from_bytes(status[6:8], "big"), tuple(f & 0xBF for f in flags), status[4], status[0], status[2])


class Controller:
    """Nonblocking operations keep keyboard handling active during movement."""

    def __init__(self, arm, receiver, home_file, identity, limits, speed=5,
                 timeout=30., emit=print):
        self.arm, self.receiver = arm, receiver
        self.home_file, self.identity = Path(home_file), identity
        self.limits = np.asarray(limits, dtype=float)
        self.speed, self.timeout, self.emit = speed, timeout, emit
        self.active = None
        self.target = None
        self.since = None
        self.anchor = None
        self.locked = False
        self.monitor = False
        self.latest = None
        self.phase = None

    def state(self):
        return decode_state(self.receiver.snapshot(), time.time())

    def stop(self, reason):
        self.active = None
        self.locked = True
        self.monitor = False
        self.since = None
        try:
            self.arm.electronic_emergency_stop()
        except Exception as error:
            self.emit(f"急停发送失败，不能确认设备已停止: {error}")
        self.emit(f"已请求电子急停，运动已锁定: {reason}。排除原因并按设备流程恢复后重启终端。")

    def validate_joints(self, joints):
        q = np.asarray(joints, dtype=float)
        if q.shape != (6,) or not np.isfinite(q).all():
            raise ValueError("初始位置必须是六个有限弧度值")
        outside = np.flatnonzero((q < self.limits[:, 0]) | (q > self.limits[:, 1]))
        if len(outside):
            details = []
            for i in outside:
                low, high = np.rad2deg(self.limits[i])
                details.append(f"第 {i + 1} 轴 {np.rad2deg(q[i]):.3f}°，允许范围 [{low:.3f}°, {high:.3f}°]")
            raise ValueError("关节越界：" + "；".join(details))
        return q

    def save_home(self, joints):
        q = self.validate_joints(joints)
        data = {"schema_version": 1, **self.identity, "joint_radians": q.tolist()}
        self.home_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.home_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(data, indent=2) + "\n")
        os.replace(temporary, self.home_file)
        self.emit(f"初始位置已保存: {self.home_file}，角度(°)={np.rad2deg(q).round(3).tolist()}")

    def load_home(self):
        data = json.loads(self.home_file.read_text())
        if data.get("schema_version") != 1 or any(data.get(k) != v for k, v in self.identity.items()):
            raise ValueError("初始位置文件的型号、CAN 接口或固件与当前会话不匹配")
        return self.validate_joints(data["joint_radians"])

    def jog_limit_deg(self, axis):
        return 20 if axis == 6 else 2

    def command(self, command):
        parts = command.split()
        jog = bool(parts and parts[0] == "jog")
        if jog:
            if len(parts) != 3:
                raise ValueError("用法: jog 轴号 增量角度；轴号 1..6，增量必须非零")
            axis, delta = int(parts[1]), float(parts[2])
            if axis not in range(1, 7) or not np.isfinite(delta) or not 0 < abs(delta) <= self.jog_limit_deg(axis):
                raise ValueError(f"轴号必须为 1..6，增量必须非零；J1–J5 最多 ±{self.jog_limit_deg(1)}°，J6 最多 ±{self.jog_limit_deg(6)}°")
            command = "jog"
        if command == "stop":
            self.stop("人工中断")
            return
        if command == "status":
            s = self.state()
            self.emit(f"角度(°)={np.rad2deg(s.joints).round(3).tolist()} 使能={s.enabled} "
                      f"状态={s.arm_status} 错误={s.error_status} 驱动故障={s.driver_faults} "
                      f"控制模式={s.ctrl_mode} 运动模式={s.mode_feedback} 到位标志={s.motion_status} "
                      f"操作={self.active} 阶段={self.phase if self.active else None} 锁定={self.locked}")
            return
        if command not in ("enable", "disable", "set-home", "zero-home", "home", "jog"):
            raise ValueError("未知命令。输入 help 查看命令。")
        if command == "disable":
            if self.active is not None:
                self.stop("失能中断当前操作")
            self.begin("disable")
            try:
                self.arm.disable()
            except BaseException:
                self.stop("失能指令发送异常")
                raise
            return
        if self.locked:
            raise RuntimeError("当前会话已锁定；仅允许 status、disable、stop 和 quit")
        if self.active:
            if command == "home" and self.active == "home":
                self.emit("正在回初始位置，不重复发送指令；Esc 可急停。")
                return
            raise RuntimeError("请等待当前操作完成；Esc 急停或 disable 失能")
        s = self.state()
        s.healthy()
        if command in ("set-home", "zero-home"):
            self.begin(command)
            self.anchor = s.joints.copy()
            self.emit("正在确认关节停稳；不会发送运动或电机零点标定指令。")
        elif command == "enable":
            self.begin("enable")
            try:
                self.arm.enable()
            except BaseException:
                self.stop("使能指令发送异常")
                raise
        else:
            if not all(s.enabled):
                raise RuntimeError("请先 enable 并等待六个关节使能完成")
            if jog:
                target = self.validate_joints(s.joints).copy()
                target[axis - 1] += np.deg2rad(delta)
                self.validate_joints(target)
            else:
                target = self.load_home()
            self.validate_joints(s.joints)
            speed = min(self.speed, 3) if jog else self.speed
            self.begin(command)
            self.target = target.copy()
            self.start_joints = s.joints.copy()
            self.phase = "等待模式确认"
            self.mode_deadline = time.monotonic() + 3.
            self.motion_speed = speed
            try:
                self.prepare_motion_mode(speed)
            except BaseException:
                self.stop("运动指令发送异常")
                raise
            action = f"微动 J{axis} {delta:+g}°" if jog else self.motion_name()
            self.destination_name = "垫子上方固定姿态" if action == "移动到垫子上方固定姿态" else "初始位置"
            self.emit(f"准备以 {speed}% 速度{action}，等待 CAN/J 模式确认，尚未发送目标；Esc/Ctrl+C 急停。")

    def prepare_motion_mode(self, speed):
        self.arm.set_speed_percent(speed)
        self.arm.set_motion_mode("j")

    def motion_name(self):
        return "回初始位置"

    def begin(self, operation):
        self.active = operation
        self.issued = time.time()
        self.deadline = time.monotonic() + self.timeout
        self.since = None
        self.anchor = None

    def pose_stable(self, sample):
        """Require distinct, fresh joint fragments across a 0.5 s interval."""
        stamp = float(np.min(sample.stamps[1:4]))
        if self.anchor is None or np.max(abs(sample.joints - self.anchor)) > np.deg2rad(.2):
            self.anchor = sample.joints.copy()
            self.since = stamp
            return False
        if self.since is None:
            self.since = stamp
        return stamp - self.since >= .5

    def tick(self):
        if self.locked and self.active != "disable":
            return
        try:
            s = self.state()
            self.latest = s
            if self.active != "disable":
                s.healthy()
            if self.active is None:
                self.monitor = self.monitor or any(s.enabled)
                return
            now = time.monotonic()
            if now > self.deadline:
                raise TimeoutError(f"{self.active} 等待反馈超时：阶段={self.phase}，"
                                   f"控制模式={s.ctrl_mode}，运动模式={s.mode_feedback}，"
                                   f"目标误差(°)={np.rad2deg(self.target - s.joints).round(3).tolist() if self.target is not None else None}")
            fresh = bool(np.all(s.stamps > self.issued))
            if self.active in ("enable", "disable"):
                desired = self.active == "enable"
                if fresh and all(v == desired for v in s.enabled):
                    if desired and not self.pose_stable(s):
                        return
                    self.monitor = desired
                    self.emit("六个关节已使能并确认停稳。" if desired else "六个关节已失能。")
                    self.active = None
                else:
                    self.anchor = None
                    self.since = None
                return
            if self.active in ("home", "jog"):
                if not all(s.enabled):
                    raise RuntimeError("运动过程中关节失能")
                self.validate_joints(s.joints)
                mode_ok = s.ctrl_mode == 1 and s.mode_feedback == 1
                if self.phase == "等待模式确认":
                    if now > self.mode_deadline:
                        raise TimeoutError(f"CAN/J 模式确认超时，未发送目标：控制模式={s.ctrl_mode}，运动模式={s.mode_feedback}")
                    if not fresh or not mode_ok:
                        self.anchor = None
                        self.since = None
                        return
                    drift = np.rad2deg(s.joints - self.start_joints)
                    if np.max(abs(drift)) > .2:
                        raise RuntimeError(f"模式确认期间实际角度变化超过 0.2°，取消目标发送：变化(°)={drift.round(3).tolist()}，到位标志={s.motion_status}")
                    if not self.pose_stable(s):
                        return
                    # Use the SDK motion transaction: mode/speed immediately
                    # followed by the three joint target frames. A prior mode
                    # feedback does not acknowledge acceptance of this target.
                    previous = self.arm.get_auto_set_motion_mode_enabled()
                    try:
                        self.arm.set_auto_set_motion_mode_enabled(True)
                        self.issued = time.time()
                        self.arm.move_j(self.target.tolist())
                    finally:
                        self.arm.set_auto_set_motion_mode_enabled(previous)
                    self.anchor = None
                    self.since = None
                    self.phase = "等待到位"
                    self.emit(f"CAN/J 模式已确认，已调用 SDK 发送模式及目标，等待执行反馈：角度(°)={np.rad2deg(self.target).round(3).tolist()}，速度={self.motion_speed}%")
                    return
                if not mode_ok:
                    raise RuntimeError(f"运动模式丢失：控制模式={s.ctrl_mode}，运动模式={s.mode_feedback}")
                tolerance = .2 if self.active == "jog" else 1
                stationary = fresh and s.motion_status == 0 and np.max(abs(s.joints - self.target)) <= np.deg2rad(tolerance)
                if self.anchor is None or np.max(abs(s.joints - self.anchor)) > np.deg2rad(.2):
                    self.anchor = s.joints.copy()
                    self.since = None
            else:
                stationary = fresh and np.max(abs(s.joints - self.anchor)) <= np.deg2rad(.2)
                if not stationary:
                    self.anchor = s.joints.copy()
            if stationary:
                self.since = now if self.since is None else self.since
                if now - self.since >= .5:
                    if self.active in ("set-home", "zero-home"):
                        target = s.joints if self.active == "set-home" else np.zeros(6)
                        # A rejected save target is not a hardware fault. Catch only
                        # validation here; feedback and I/O failures retain stop handling.
                        try:
                            self.validate_joints(target)
                        except ValueError as error:
                            self.active = None
                            self.anchor = None
                            self.since = None
                            self.monitor = self.monitor or any(s.enabled)
                            self.emit(f"拒绝保存初始位置：{error}。原文件未修改；可调整姿态后重试。")
                            return
                        self.save_home(target)
                    else:
                        self.emit("单关节微动已完成并停稳。" if self.active == "jog" else f"已到达{self.destination_name}并停稳。")
                    self.active = None
            else:
                self.since = None
        except Exception as error:
            if self.active is not None or self.monitor:
                self.stop(str(error))


HELP = """命令（输入后回车）：
  status     查看关节角、使能和故障状态
  set-home   停稳后保存当前姿态为初始位置（不修改电机零点）
  zero-home  将初始位置设为六关节 0°（仅保存，不运动）
  enable     使能六个关节
  home       回到保存的初始位置
  jog 6 2    指定关节相对微动（轴号 1..6，J1–J5 ±2°、J6 ±20° 内，非零，速度最多 3%）
  disable    失能；请先支撑机械臂，失能后可能下落
  stop       电子急停并锁定本次会话
  quit       退出；运动中退出会请求急停，静止时不自动失能
快捷键（无需回车）：Esc = 急停；Ctrl+C = 急停并退出。
电子急停为阻尼停止，机械臂可能缓慢下沉；通信中断时不能保证送达。
回初始位置是关节空间运动，不是避障或故障恢复。须确保整个路径无障碍。
快捷键仅作用于本终端，不控制其他进程运行的抓取程序。"""


@contextmanager
def keyboard():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        # Ctrl+S must not freeze output and block the keyboard/feedback loop.
        attributes = termios.tcgetattr(fd)
        attributes[0] &= ~(termios.IXON | termios.IXOFF)
        termios.tcsetattr(fd, termios.TCSANOW, attributes)
        yield fd
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def console(controller):
    print(HELP, flush=True)
    pending = ""
    with keyboard() as fd:
        while True:
            controller.tick()
            if not select.select([fd], [], [], .02)[0]:
                continue
            key = os.read(fd, 1).decode("utf-8", errors="ignore")
            if not key or key == "\x04":
                return
            if key == "\x1b":
                pending = ""
                controller.stop("Esc")
                continue
            if key in ("\r", "\n"):
                print()
                command, pending = pending.strip(), ""
            elif key in ("\x7f", "\b"):
                pending = pending[:-1]
                print("\b \b", end="", flush=True)
                continue
            else:
                if key.isprintable() and len(pending) < 80:
                    pending += key
                    print(key, end="", flush=True)
                continue
            if command == "quit":
                return
            if command == "help":
                print(HELP, flush=True)
                continue
            if command:
                try:
                    controller.command(command)
                except Exception as error:
                    print(f"拒绝执行: {error}", flush=True)


@contextmanager
def shutdown_signals():
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    previous = {}
    try:
        # SSH hangup and terminal suspension must unwind through stop/cleanup.
        for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGTSTP):
            previous[signum] = signal.signal(signum, interrupted)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def add_arguments(p):
    p.add_argument("--channel", default="can0")
    p.add_argument("--home-file", type=Path, default=Path("local/arm_home.json"))
    p.add_argument("--speed", type=int, choices=range(1, 21), default=5,
                   metavar="1..20", help="回初始位置速度百分比，默认 5")


def main(argv=None):
    p = argparse.ArgumentParser(description="Piper X 使能、保存初始位置、回零和失能终端")
    add_arguments(p)
    args = p.parse_args(argv)
    if not sys.stdin.isatty():
        p.error("需要交互终端，以便运动过程中接收 Esc；不支持管道输入")
    with shutdown_signals():
        return run_session(args)


def run_session(args):
    from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD

    def create(profile):
        return AgxArmFactory.create_arm(create_agx_arm_config(
            robot="piper_x", firmeware_version=profile,
            interface="socketcan", channel=args.channel))

    arm = receiver = controller = None
    try:
        arm = create("default")
        arm.connect()
        firmware = arm.get_firmware(timeout=3)
        if firmware is None:
            raise RuntimeError("固件查询超时，请检查 CAN 和机械臂供电")
        version = firmware["software_version"]
        profile = resolve_firmware_profile("piper_x", version)
        arm.disconnect()
        arm = create(profile)
        receiver = JointReceiver(args.channel)
        arm.connect()
        limits = list(ROBOT_JOINT_LIMIT_PRESET_RAD["piper_x"].values())
        controller = Controller(arm, receiver, args.home_file,
                                {"model": "piper_x", "channel": args.channel, "firmware": version},
                                limits, speed=args.speed)
        print(f"已连接 Piper X，固件 {version}，初始位置文件 {args.home_file.resolve()}。未自动使能。")
        deadline = time.monotonic() + 3
        while True:
            try:
                state = controller.state()
                controller.monitor = any(state.enabled)
                break
            except RuntimeError:
                if time.monotonic() > deadline:
                    raise
                time.sleep(.02)
        console(controller)
        return 0
    except KeyboardInterrupt:
        if controller is not None:
            controller.stop("Ctrl+C、终止、SSH 断连或终端挂起信号")
        return 130
    finally:
        try:
            if controller is not None and controller.active is not None:
                controller.stop("操作未完成时退出")
        finally:
            try:
                if receiver is not None:
                    receiver.close()
            finally:
                if arm is not None:
                    arm.disconnect()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
