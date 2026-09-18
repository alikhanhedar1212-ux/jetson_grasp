"""One-shot Piper X emergency-stop recovery; never enable or move the arm."""
import argparse
import time

import numpy as np

from .arm_console import JointReceiver, decode_state


def check_state(sample):
    if any(sample.enabled):
        raise RuntimeError("必须六轴失能；请先支撑好机械臂并失能")
    if sample.error_status or any(sample.driver_faults) or sample.arm_status not in (0, 1):
        raise RuntimeError("存在其他故障，拒绝复位")


def recover(arm, receiver, timeout=3., emit=print):
    def read():
        sample = decode_state(receiver.snapshot(), time.time())
        check_state(sample)
        return sample

    first = read()
    for _ in range(10):
        time.sleep(.05)
        current = read()
        if np.max(abs(current.joints - first.joints)) > np.deg2rad(.2):
            raise RuntimeError("姿态未停稳，拒绝复位")
    if current.arm_status == 0:
        emit("设备已正常，无需复位；六轴仍未使能")
        return
    issued = time.time()
    arm.reset()
    emit("已发送一次 SDK reset，正在等待新反馈；不会自动重试")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        time.sleep(.05)
        current = read()
        if np.all(current.stamps > issued) and current.arm_status == 0:
            emit("急停已解除：状态=0，六轴仍未使能；未发送运动指令")
            return
    raise RuntimeError("未确认复位成功；不要继续使能或重复复位，请检查反馈")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", default="can1")
    args = parser.parse_args(argv)
    from pyAgxArm import AgxArmFactory, create_agx_arm_config, resolve_firmware_profile

    arm = receiver = None
    try:
        print("请确保已退出其他控制终端、机械臂已支撑且运动区域清空。", flush=True)
        receiver = JointReceiver(args.channel)
        arm = AgxArmFactory.create_arm(create_agx_arm_config(
            robot="piper_x", firmeware_version="default",
            interface="socketcan", channel=args.channel))
        arm.connect()
        time.sleep(.5)
        check_state(decode_state(receiver.snapshot(), time.time()))
        firmware = arm.get_firmware(timeout=5)
        if firmware is None:
            raise RuntimeError("固件查询超时，未发送复位")
        version = firmware["software_version"]
        profile = resolve_firmware_profile("piper_x", version)
        arm.disconnect()
        arm = AgxArmFactory.create_arm(create_agx_arm_config(
            robot="piper_x", firmeware_version=profile,
            interface="socketcan", channel=args.channel))
        arm.connect()
        print(f"接口={args.channel}，固件={version}，驱动={profile}")
        recover(arm, receiver)
        return 0
    except (Exception, KeyboardInterrupt) as error:
        print(f"恢复未完成: {error}。保持停机，勿自动重试。", flush=True)
        return 2
    finally:
        try:
            if receiver is not None:
                receiver.close()
        finally:
            if arm is not None:
                arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
