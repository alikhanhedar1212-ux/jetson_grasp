"""Fixed-pose J6 sweep with timestamped CAN and explicit FK model comparisons."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from . import arm_console as ac
from .mat_pose import FixedController, TARGET_DEG, poll_key
from .robot import decode_feedback
from .geometry import pose_matrix, pose_error


class Receiver(ac.JointReceiver):
    IDS = ac.JointReceiver.IDS + (0x2A2, 0x2A3, 0x2A4)


class Controller(FixedController):
    experiment_target = None
    fixed_anchor = None

    def load_home(self):
        return self.validate_joints(self.experiment_target).copy()

    def motion_name(self):
        return '移动到实验关节目标'

    def tick(self):
        super().tick()
        if self.fixed_anchor is not None and not self.locked:
            try:
                s = self.state()
                s.healthy()
                self.validate_joints(s.joints)
                if not all(s.enabled) or s.ctrl_mode != 1 or s.mode_feedback != 1:
                    raise RuntimeError('实验期间使能或 CAN/J 模式丢失')
                if np.max(abs(s.joints[:5] - self.fixed_anchor[:5])) > np.deg2rad(.2):
                    raise RuntimeError('J1–J5 偏离固定基准超过 0.2°，单变量条件失效')
            except Exception as error:
                self.stop(str(error))


def make_targets(base_deg, offsets_deg, limits):
    base = np.deg2rad(np.asarray(base_deg, dtype=float))
    offsets = np.asarray(offsets_deg, dtype=float)
    if base.shape != (6,) or not np.isfinite(base).all():
        raise ValueError('固定点必须包含六个有限角度')
    if offsets.ndim != 1 or len(offsets) < 2 or not np.isfinite(offsets).all():
        raise ValueError('至少需要两个有限 J6 偏移值')
    if offsets[0] != 0 or offsets[-1] != 0:
        raise ValueError('偏移序列必须以 0 开始和结束')
    if np.max(abs(np.diff(offsets))) > 20:
        raise ValueError('相邻 J6 实验点的步长不得超过 20°')
    targets = np.tile(base, (len(offsets), 1))
    targets[:, 5] += np.deg2rad(offsets)
    bounds = np.asarray(limits)
    if np.any(targets < bounds[:, 0]) or np.any(targets > bounds[:, 1]):
        raise ValueError('固定点或 J6 扫描目标超过 SDK 关节限位')
    return targets


def snapshot(receiver, now):
    frames = receiver.snapshot()
    state = ac.decode_state(frames, now)
    state.healthy()
    pose = decode_feedback(frames, np.eye(4))
    stamps = np.array([frames[i][1] for i in ac.JOINT_IDS + (0x2A2, 0x2A3, 0x2A4)])
    if not np.isfinite(stamps).all() or np.any(now - stamps < 0) or np.any(now - stamps > .25):
        raise RuntimeError('关节/法兰反馈过期')
    if np.ptp(stamps) > .05:
        raise RuntimeError('关节/法兰分帧时间差超过 50 ms')
    return state, pose.tcp, stamps


def record(receiver, index, offset, target, fk):
    state, can_pose, stamps = snapshot(receiver, time.time())
    result = dict(index=index, offset_deg=float(offset), host_wall_s=time.time(),
                  target_joint_rad=target.tolist(), joint_rad=state.joints.tolist(),
                  T_base_flange_can=can_pose.tolist(), component_stamps_s=stamps.tolist())
    for model in ('piper', 'piper_x'):
        matrix = pose_matrix(fk(model, state.joints.tolist()))
        distance, angle = pose_error(can_pose, matrix)
        result[model] = dict(T_base_flange=matrix.tolist(),
                             can_error_mm=distance * 1000, can_error_deg=float(np.rad2deg(angle)))
    return result


def run_sequence(controller, targets, offsets, dwell, poll, save, fk, home_target=None, observe=None):
    def step():
        poll()
        controller.tick()
        if controller.locked:
            raise RuntimeError('实验停止，禁止继续发送目标')

    def wait_active():
        while controller.active is not None:
            step()
            time.sleep(.02)

    try:
        # Validate the entire plan before even enabling.
        for target in targets:
            controller.validate_joints(target)
        if home_target is not None:
            controller.validate_joints(home_target)
        poll()
        controller.command('enable')
        wait_active()
        for index, (target, offset) in enumerate(zip(targets, offsets)):
            poll()
            controller.experiment_target = target.copy()
            controller.command('home')
            wait_active()
            # Refine the inherited 1-degree arrival check to 0.2 degrees;
            # require fresh distinct joint+pose feedback throughout dwell.
            deadline = time.monotonic() + controller.timeout
            since = None
            while True:
                step()
                state, _, stamps = snapshot(controller.receiver, time.time())
                stationary = (state.motion_status == 0 and np.all(stamps > controller.issued)
                              and np.max(abs(state.joints - target)) <= np.deg2rad(.2))
                if stationary:
                    since = float(stamps.min()) if since is None else since
                    if float(stamps.min()) - since >= dwell:
                        break
                else:
                    since = None
                if time.monotonic() > deadline:
                    raise TimeoutError('实验点未在限定时间内到位并停稳')
                time.sleep(.02)
            if index == 0:
                # Freeze the measured first-five angles to avoid correcting
                # their small initial arrival errors during the J6 sweep.
                controller.fixed_anchor = state.joints.copy()
                targets[:, :5] = state.joints[:5]
            sample = record(controller.receiver, index, offset, target, fk)
            if observe is not None:
                observation = observe(sample, target.copy(), step)
                sample['camera_observation'] = dict(valid=observation['valid'],
                    path=f"sample_{index:04d}/observation.json", error=observation.get('error'))
                controller.emit('D405 棋盘观测：' + ('有效' if observation['valid'] else observation['error']))
            save(sample)
            controller.emit(f'记录点 {index}: J6 偏移 {offset:+g}°，实际 J6={np.rad2deg(state.joints[5]):.3f}°')
        if home_target is not None:
            poll()
            controller.fixed_anchor = None
            controller.experiment_target = np.asarray(home_target).copy()
            controller.emit('J6 采样完成，开始以 3% 速度返回已保存的 home 姿态。')
            controller.command('home')
            wait_active()
    except BaseException:
        if not controller.locked:
            controller.stop('J6 实验中断或失败；不自动返回')
        raise


def query_firmware(arm):
    """Wait for SDK receive FPS before requesting segmented firmware data."""
    deadline = time.monotonic() + 5
    while arm.get_fps() <= 0:
        if time.monotonic() >= deadline:
            raise RuntimeError('连接后 5 秒未建立 CAN 接收帧率；检查供电、接口和反馈')
        time.sleep(.05)
    for attempt in range(2):
        firmware = arm.get_firmware(timeout=5)
        if firmware is not None:
            return firmware
        if attempt == 0:
            time.sleep(1)
    raise RuntimeError('CAN 接收已就绪，但两次固件查询均失败；未请求使能或运动')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel', default='can1')
    parser.add_argument('--home-file', type=Path, default=Path(__file__).resolve().parents[1] / 'local/arm_home.json')
    parser.add_argument('--base-deg', nargs=6, type=float, default=TARGET_DEG, metavar='DEG')
    parser.add_argument('--offsets-deg', nargs='+', type=float, default=[0, 5, 10, 5, 0, -5, -10, -5, 0])
    parser.add_argument('--dwell', type=float, default=2., help='每点连续停稳秒数，至少 0.5')
    parser.add_argument('--output', type=Path, default=Path('runs') / time.strftime('j6_%Y%m%d_%H%M%S'))
    parser.add_argument('--d405', action='store_true', help='每点采集 D405 固定棋盘')
    parser.add_argument('--handeye', type=Path, default=Path(__file__).resolve().parents[2] / 'data/handeye/d405_01_fixed_split/candidate_result.json')
    parser.add_argument('--execute', action='store_true', help='实际连接并运动；默认只打印计划')
    args = parser.parse_args(argv)
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    limits = list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values())
    try:
        targets = make_targets(args.base_deg, args.offsets_deg, limits)
        if not np.isfinite(args.dwell) or args.dwell < .5 or args.dwell > 20:
            raise ValueError('dwell 必须在 0.5–20 秒之间')
    except ValueError as error:
        parser.error(str(error))
    print(f'固定点(°): {list(args.base_deg)}；速度 3%；J6 相对固定点偏移(°): {args.offsets_deg}')
    print(f'先到固定点，再扫描 J6，最后返回 home：{args.home_file}；不操作夹爪。')
    observer = None
    if args.d405:
        from .j6_camera import Observation
        observer = Observation(args.output, args.handeye)
        print('D405：9×6 内角点、20 mm；每点保存 RGB-D 和棋盘，扫描结束整臂回 home。')
    if not args.execute:
        print('计划预览完成，未连接 CAN。添加 --execute 执行。')
        return 0
    if not sys.stdin.isatty():
        parser.error('实际执行需要交互终端，以接收空格/Esc 急停')
    args.output.mkdir(parents=True, exist_ok=False)
    from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile
    def create(profile):
        return AgxArmFactory.create_arm(create_agx_arm_config(robot='piper_x',
            firmeware_version=profile, interface='socketcan', channel=args.channel))
    mdh = {name: get_mdh(name) for name in ('piper', 'piper_x')}
    def fk(name, joints):
        return fk_from_mdh(mdh[name], joints)
    arm = receiver = controller = None
    try:
        with ac.shutdown_signals():
            arm = create('default')
            arm.connect()
            firmware = query_firmware(arm)
            profile = resolve_firmware_profile('piper_x', firmware['software_version'])
            arm.disconnect()
            arm = create(profile)
            receiver = Receiver(args.channel)
            arm.connect()
            controller = Controller(arm, receiver, args.home_file,
                {'model': 'piper_x', 'channel': args.channel, 'firmware': firmware['software_version']},
                limits, speed=3)
            home_target = ac.Controller.load_home(controller).copy()
            time.sleep(.5)
            snapshot(receiver, time.time())
            if observer is not None:
                observer.start(receiver)
            metadata = dict(base_deg=list(args.base_deg), offsets_deg=args.offsets_deg,
                            dwell_s=args.dwell, speed_percent=3, channel=args.channel,
                            firmware=firmware, driver_model='piper_x', fk_models=list(mdh),
                            camera=None if observer is None else observer.info,
                            mdh_m_rad=mdh, home_joint_rad=home_target.tolist(), completed=False)
            meta_path = args.output / 'metadata.json'
            meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
            print('须从六轴失能且停稳状态开始。清空到固定点的整段路径及腕部、相机和线缆旋转空间。')
            print('空格/Esc/Ctrl+C 请求急停；故障后不会自动回程。', flush=True)
            with (args.output / 'samples.jsonl').open('x') as output, ac.keyboard() as fd:
                def save(sample):
                    output.write(json.dumps(sample) + '\n')
                    output.flush()
                run_sequence(controller, targets, args.offsets_deg, args.dwell,
                             lambda: poll_key(fd), save, fk, home_target=home_target,
                             observe=None if observer is None else observer.collect)
            if observer is not None:
                observer.report()
            metadata['completed'] = True
            meta_path.write_text(json.dumps(metadata, indent=2) + '\n')
            print(f'实验完成，整臂已返回保存的 home 并停稳；保持使能，不自动失能。记录：{args.output.resolve()}')
            return 0
    except BaseException as error:
        if controller is not None and not controller.locked:
            controller.stop(f'实验退出: {error}')
        print(f'实验未完成: {error}', file=sys.stderr)
        return 2
    finally:
        try:
            if observer is not None:
                observer.close()
            if receiver is not None:
                receiver.close()
        finally:
            if arm is not None:
                arm.disconnect()


if __name__ == '__main__':
    raise SystemExit(main())
