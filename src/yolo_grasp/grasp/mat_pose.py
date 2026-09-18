"""Move to the user-taught mat pose, then open the AGX gripper."""
import argparse
import math
import os
from pathlib import Path
import select
import sys
import time

import numpy as np
from . import arm_console as ac

TARGET_DEG = (-5.0, 56.138, -52.908, -2.0, 38.796, -0.902)


class Receiver(ac.JointReceiver):
    IDS = ac.JointReceiver.IDS + (0x2A8,)


class UserStop(Exception):
    """Cancel automatic work while keeping the terminal available."""


class FixedController(ac.Controller):
    mat_target = False
    prepared = False

    def motion_name(self):
        return '移动到垫子上方固定姿态' if self.mat_target else '回初始位置'

    def prepare_motion_mode(self, speed):
        if not self.prepared or speed != 3:
            raise RuntimeError('请先从六轴失能状态执行本程序的 enable 准备流程')
        sample = self.state()
        if sample.ctrl_mode != 1 or sample.mode_feedback != 1:
            raise RuntimeError('已准备的 CAN/J 模式丢失，拒绝带使能重新切换模式')

    def adopt_enabled_session(self):
        """Accept a session that already runs enabled in CAN/J mode.

        The disable -> enable preparation cannot run while the axes are already
        enabled, which used to leave `home` unreachable in exactly that state.
        The 3% speed is seeded into the SDK's cached mode message instead of
        being sent as a standalone frame (that frame leaves mode_feedback at
        0xFF on this firmware), so the final move_j transaction still carries
        mode, speed and targets together.
        """
        sample = self.state()
        sample.healthy()
        self.validate_joints(sample.joints)
        if not all(sample.enabled):
            raise RuntimeError('六轴未全部使能，不能采用当前会话')
        if sample.ctrl_mode != 1 or sample.mode_feedback != 1:
            raise RuntimeError('已使能但不是 CAN/J 模式，拒绝采用当前会话')
        if self.speed != 3:
            raise RuntimeError('本入口固定 3% 速度')
        mode = getattr(self.arm, '_msg_mode', None)
        if mode is None or not hasattr(mode, 'move_spd_rate_ctrl'):
            raise RuntimeError('SDK 未暴露可缓存的速度字段；拒绝以默认速度回位')
        mode.move_spd_rate_ctrl = 3
        self.prepared = True
        return sample

    def enable_needed(self):
        """True only while the session still needs the disable -> enable prep."""
        return not self.prepared
        # Speed was prepared while disabled; do not issue an early standalone
        # mode command here. The final move_j transaction sends mode + target.

    def tick(self):
        if self.active != 'enable' or self.phase not in ('失能停稳检查', '失能目标准备'):
            return super().tick()
        try:
            sample = self.state()
            sample.healthy()
            self.validate_joints(sample.joints)
            if any(sample.enabled):
                raise RuntimeError('准备期间出现关节使能，停止准备')
            if time.monotonic() > self.deadline:
                raise TimeoutError('失能准备超时，未请求使能')
            if not np.all(sample.stamps > self.issued):
                return
            if self.phase == '失能目标准备':
                if time.monotonic() > self.mode_deadline:
                    raise TimeoutError('失能状态下 CAN/J 模式确认超过 3 秒，未请求使能')
                drift = np.rad2deg(sample.joints - self.start_joints)
                if np.max(abs(drift)) > .2:
                    raise RuntimeError(f'失能准备期间姿态变化超过 0.2°：{drift.round(3).tolist()}')
                if sample.ctrl_mode != 1 or sample.mode_feedback != 1:
                    self.anchor = None
                    self.since = None
                    return
            if not self.pose_stable(sample):
                return
            if self.phase == '失能停稳检查':
                self.start_joints = sample.joints.copy()
                # Seed the holding target before switching mode or enabling.
                previous = self.arm.get_auto_set_motion_mode_enabled()
                try:
                    self.arm.set_auto_set_motion_mode_enabled(False)
                    self.arm.move_j(self.start_joints.tolist())
                finally:
                    self.arm.set_auto_set_motion_mode_enabled(previous)
                self.arm.set_speed_percent(3)
                self.arm.set_motion_mode('j')
                self.issued = time.time()
                self.anchor = self.since = None
                self.mode_deadline = time.monotonic() + 3
                self.phase = '失能目标准备'
                self.emit('六轴仍失能：已发送当前位置保持目标及 3% CAN/J 模式，等待新反馈；未请求使能。')
            else:
                self.prepared = True
                self.phase = '等待使能停稳'
                self.issued = time.time()
                self.anchor = self.since = None
                self.arm.enable()
                self.emit('失能状态及模式反馈已确认，已请求使能；等待停稳。')
        except BaseException as error:
            self.prepared = False
            self.stop(f'失能准备失败：{error}；不发送垫子目标或张爪指令')
            raise

    def load_home(self):
        if self.mat_target:
            return self.validate_joints(np.deg2rad(TARGET_DEG)).copy()
        return super().load_home()

    def move_to_mat(self):
        self.mat_target = True
        try:
            super().command('home')
        finally:
            self.mat_target = False

    def command(self, command):
        if command not in ('enable', 'disable', 'home', 'status', 'stop'):
            raise ValueError('此程序仅允许 status、disable、stop、quit')
        if command == 'disable':
            self.prepared = False
        if command == 'enable':
            if self.locked or self.active:
                raise RuntimeError('会话锁定或操作未完成，拒绝使能')
            sample = self.state()
            sample.healthy()
            self.validate_joints(sample.joints)
            if any(sample.enabled):
                raise RuntimeError('请先支撑机械臂并 disable，需从六轴失能状态准备')
            self.prepared = False
            self.begin('enable')
            self.phase = '失能停稳检查'
            self.emit('先确认六轴失能且停稳，再准备当前位置保持目标和模式。')
            return
        return super().command(command)


def gripper_state(receiver):
    frame = receiver.snapshot().get(0x2A8)
    if frame is None:
        raise RuntimeError('缺少夹爪反馈')
    data, stamp = frame
    if len(data) != 8 or not 0 <= time.time() - stamp <= .25:
        raise RuntimeError('夹爪反馈过期或长度错误')
    if data[7] != 0 or data[6] & 0xBF:
        raise RuntimeError('夹爪非宽度模式、存在故障或回零状态位为 1')
    width = int.from_bytes(data[:4], 'big', signed=True) * 1e-6
    if width < 0:
        raise RuntimeError('夹爪反馈开度为负')
    if width > .15:
        raise RuntimeError(f'夹爪反馈开度异常（{width*1000:.1f} mm，超过最大行程；非宽度模式或错帧）')
    return width, stamp, bool(data[6] & 0x40)


def configured_max_width(gripper):
    feedback = gripper.get_gripper_teaching_pendant_param(timeout=3)
    if feedback is None:
        raise RuntimeError('最大行程查询超时，未执行动作')
    width = float(feedback.msg.max_range_config)
    if not math.isfinite(width) or not any(abs(width - v) < 1e-6 for v in (.07, .1)):
        raise RuntimeError(f'最大行程配置无效: {width}，未执行动作')
    return width


def check_opening(receiver, width):
    actual, _, _ = gripper_state(receiver)
    if width < actual:
        raise ValueError(f'目标 {width * 1000:g} mm 小于当前 {actual * 1000:g} mm，拒绝闭合夹爪')


def run_sequence(controller, gripper, width, force, poll):
    """poll is nonblocking keyboard handling; no automatic retries or recovery."""
    if not math.isfinite(width) or width <= 0 or not math.isfinite(force) or not 0 < force <= 32.767:
        raise ValueError('开度必须为正有限值；力必须为正且可由 SDK 反馈字段表示')
    state = controller.state()
    state.healthy()
    controller.validate_joints(state.joints)
    controller.validate_joints(np.deg2rad(TARGET_DEG))
    check_opening(controller.receiver, width)
    opening_sent = False
    try:
        for command in (('home',) if controller.prepared else ('enable', 'home')):
            poll()
            if command == 'home':
                controller.move_to_mat()
            else:
                controller.command(command)
            while controller.active is not None:
                poll()
                controller.tick()
                if controller.locked:
                    raise RuntimeError('机械臂操作失败；不张开夹爪')
                time.sleep(.02)
        # The home controller allows 1 degree arrival tolerance. Confirm the
        # same bounded pose throughout opening, in addition to arm health.
        check_opening(controller.receiver, width)
        anchor = controller.state().joints.copy()
        issued = time.time()
        opening_sent = True
        gripper.move_gripper_m(value=width, force=force)
        controller.emit(f'已到达垫子上方；张开夹爪至 {width * 1000:g} mm，力参数 {force:g} N')
        deadline = time.monotonic() + 10
        since = None
        while time.monotonic() < deadline:
            poll()
            controller.tick()
            if controller.locked:
                raise RuntimeError('机械臂监控触发停止')
            state = controller.state()
            state.healthy()
            controller.validate_joints(state.joints)
            if not all(state.enabled) or state.ctrl_mode != 1 or state.mode_feedback != 1:
                raise RuntimeError('张爪期间机械臂失能或控制模式变化')
            if np.max(abs(state.joints - anchor)) > np.deg2rad(.2):
                raise RuntimeError('张爪期间机械臂姿态变化超过 0.2°')
            actual, stamp, enabled = gripper_state(controller.receiver)
            if stamp > issued and enabled and abs(actual - width) <= .001:
                since = stamp if since is None else since
                if stamp - since >= .5:
                    controller.emit('固定姿态及张爪完成，继续保持使能。支撑好机械臂后可输入 disable；quit 不自动失能。')
                    return
            else:
                since = None
            time.sleep(.02)
        raise TimeoutError('夹爪未在 10 秒内确认开度并停稳')
    except BaseException:
        if not controller.locked:
            controller.stop('固定姿态/张爪流程中断或失败')
        if opening_sent:
            try:
                gripper.disable_gripper()
            except Exception as error:
                controller.emit(f'夹爪失能请求失败: {error}')
        raise


def poll_key(fd):
    if select.select([fd], [], [], 0)[0]:
        key = os.read(fd, 1)
        if key in (b' ', b'\x1b'):
            raise UserStop('空格/Esc：取消流程并请求电子急停')
        if key in (b'', b'\x04', b'\x03'):
            raise KeyboardInterrupt('Esc/Ctrl+C/终端关闭')
        # No buffered commands from the automatic sequence carry into hold mode.


def hold(controller, fd):
    print('保持监控：status / enable / disable / home / stop / quit；空格或 Esc 急停。急停后须先按流程恢复，home 不复位。')
    pending = ''
    while True:
        controller.tick()
        if not select.select([fd], [], [], .02)[0]:
            continue
        key = os.read(fd, 1)
        if key in (b'', b'\x04'):
            return
        if key in (b' ', b'\x1b'):
            pending = ''
            controller.stop('空格/Esc')
        elif key in (b'\r', b'\n'):
            print()
            command, pending = pending.strip(), ''
            if command == 'quit':
                return
            if command not in ('status', 'enable', 'disable', 'home', 'stop'):
                print('仅支持 status / enable / disable / home / stop / quit')
                continue
            try:
                controller.command(command)
            except Exception as error:
                print(f'拒绝执行: {error}')
        elif key in (b'\x7f', b'\b'):
            pending = pending[:-1]
            print('\b \b', end='', flush=True)
        else:
            text = key.decode('utf8', errors='ignore')
            if text.isprintable() and len(pending) < 80:
                pending += text
                print(text, end='', flush=True)


def run_fixed_only(controller, poll):
    """Prepare disabled arm, move to fixed pose, and verify fresh settled feedback."""
    controller.validate_joints(np.deg2rad(TARGET_DEG))
    actions = ([controller.move_to_mat] if controller.prepared
               else [lambda: controller.command('enable'), controller.move_to_mat])
    for action in actions:
        poll()
        action()
        while controller.active is not None:
            poll(); controller.tick()
            if controller.locked:
                raise RuntimeError('固定点准备已停止')
            time.sleep(.02)
    deadline = time.monotonic() + controller.timeout
    since = None
    while time.monotonic() < deadline:
        poll(); controller.tick()
        if controller.locked:
            raise RuntimeError('固定点监控已停止')
        state = controller.state()
        state.healthy()
        if not all(state.enabled) or state.ctrl_mode != 1 or state.mode_feedback != 1:
            raise RuntimeError('固定点使能或CAN/J模式丢失')
        fresh = np.all(state.stamps > controller.issued)
        if fresh and state.motion_status == 0 and np.max(abs(state.joints-np.deg2rad(TARGET_DEG))) <= np.deg2rad(.2):
            stamp = float(np.min(state.stamps))
            since = stamp if since is None else since
            if stamp-since >= .5:
                controller.emit('固定抓取点已到位：六轴误差≤0.2°，连续停稳0.5秒；保持使能。输入 quit 退出后可启动三轴测试。')
                return
        else:
            since = None
        time.sleep(.02)
    raise TimeoutError('固定点未确认0.2°精度及连续停稳')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='can1')
    parser.add_argument('--home-file', type=Path, default=Path(__file__).resolve().parents[1] / 'local/arm_home.json')
    parser.add_argument('--home-only', action='store_true', help='仅进入控制终端，不执行垫子移动或张爪')
    parser.add_argument('--fixed-only', action='store_true', help='只使能并到固定点，不操作夹爪')
    opening = parser.add_mutually_exclusive_group()
    opening.add_argument('--open-max', action='store_true', help='读取设备最大行程配置作为全开目标')
    opening.add_argument('--open-mm', type=float, help='已核实适合当前夹爪的张开宽度，毫米')
    parser.add_argument('--force-n', type=float, help='已核实适合当前夹爪的力参数，牛顿')
    args = parser.parse_args(argv)
    if args.fixed_only and (args.home_only or args.open_max or args.open_mm is not None or args.force_n is not None):
        parser.error('--fixed-only 不能与 --home-only 或夹爪参数同时使用')
    if not args.home_only and not args.fixed_only and (not args.open_max and args.open_mm is None or args.force_n is None):
        parser.error('自动流程需要 --open-max 或 --open-mm，以及 --force-n')
    if args.open_mm is not None and (not math.isfinite(args.open_mm) or not 0 < args.open_mm <= 2147483):
        parser.error('开度必须为正有限值且可由 SDK 编码；不得超过实际夹爪行程')
    if args.force_n is not None and (not math.isfinite(args.force_n) or not 0 < args.force_n <= 32.767):
        parser.error('力参数必须为正且不超过 SDK 反馈字段范围 32.767 N；还须符合夹爪规格')
    if not sys.stdin.isatty():
        parser.error('需要交互终端以接收 Esc 急停')
    from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    def create(profile):
        return AgxArmFactory.create_arm(create_agx_arm_config(
            robot='piper_x', firmeware_version=profile, interface='socketcan', channel=args.channel))
    arm = receiver = controller = None
    with ac.shutdown_signals():
        try:
            arm = create('default')
            arm.connect()
            time.sleep(.5)
            firmware = arm.get_firmware(timeout=5)
            if firmware is None:
                raise RuntimeError('固件查询失败，未执行动作')
            profile = resolve_firmware_profile('piper_x', firmware['software_version'])
            arm.disconnect()
            arm = create(profile)
            receiver = Receiver(args.channel)
            arm.connect()
            gripper = None if args.fixed_only else arm.init_effector(arm.OPTIONS.EFFECTOR.AGX_GRIPPER)
            width = None if args.home_only or args.fixed_only else (configured_max_width(gripper) if args.open_max else args.open_mm / 1000)
            controller = FixedController(arm, receiver, args.home_file,
                {'model': 'piper_x', 'channel': args.channel, 'firmware': firmware['software_version']},
                list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values()), speed=3)
            # Never access or overwrite the user's home file.
            time.sleep(.5)
            state = controller.state()
            if all(state.enabled):
                controller.adopt_enabled_session()
                print('六轴已使能且为 CAN/J 模式：已采用当前会话（3% 速度只随目标帧一起发送）；'
                      'home 可直接使用。', flush=True)
            if width is not None:
                print(f'固定目标(°)={list(TARGET_DEG)}；速度=3%；开度={width * 1000:g} mm')
            print('空格/Esc 急停；执行运动前须清空整段路径、末端及线缆空间。', flush=True)
            with ac.keyboard() as fd:
                if args.fixed_only:
                    run_fixed_only(controller, lambda: poll_key(fd))
                elif not args.home_only:
                    try:
                        run_sequence(controller, gripper, width, args.force_n, lambda: poll_key(fd))
                    except UserStop as error:
                        print(f'{error}。自动流程已取消，未执行的张爪不会继续。')
                hold(controller, fd)
            return 0
        except BaseException as error:
            if controller is not None and not controller.locked and (controller.active or controller.monitor):
                controller.stop(f'程序退出: {error}')
            print(f'流程未完成: {error}', flush=True)
            return 2
        finally:
            try:
                if controller is not None and controller.active is not None and not controller.locked:
                    controller.stop('操作中退出')
            finally:
                try:
                    if receiver is not None:
                        receiver.close()
                finally:
                    if arm is not None:
                        arm.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
