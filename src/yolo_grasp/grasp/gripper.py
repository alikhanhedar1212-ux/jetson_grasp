"""AGX gripper closing with width/force feedback.

This module never moves the arm. Closing is commanded in small width steps and
stops when the jaws stop following the command, which is what "closed until it
cannot close further" means with a block between them. The 0x2A8 feedback is
read through its own receive-only socket so the arm state decoder is untouched.
"""
import struct
import threading
import time

GRIPPER_FRAME_ID = 0x2A8


def decode_gripper_frame(frame, now, max_age=.25):
    """Validate one 0x2A8 frame; width-mode, fault-free and fresh only."""
    if frame is None:
        raise RuntimeError('缺少夹爪反馈')
    data, stamp = frame
    if not isinstance(data, (bytes, bytearray)) or len(data) != 8:
        raise RuntimeError('夹爪反馈长度错误')
    if not 0. <= now - float(stamp) <= max_age:
        raise RuntimeError(f'夹爪反馈过期或时钟异常（{now - float(stamp):+.3f} s）')
    if data[7] != 0 or data[6] & 0xBF:
        raise RuntimeError('夹爪非宽度模式、存在故障或回零状态位为 1')
    width = int.from_bytes(data[:4], 'big', signed=True) * 1e-6
    force = int.from_bytes(data[4:6], 'big', signed=True) * 1e-3
    if width < 0:
        raise RuntimeError('夹爪反馈开度为负')
    # A frame can carry the width-mode bits yet decode garbage (2026-09-14: the
    # first "ready" frame reported 16777.5 mm and the close was refused mid-loop).
    # Bound it by the real stroke so the wait keeps looking instead of trusting it.
    if width > .15:
        raise RuntimeError(f'夹爪反馈开度异常（{width*1000:.1f} mm，超过最大行程；非宽度模式或错帧）')
    return dict(width_m=width, force_n=force, active=bool(data[6] & 0x40), stamp=float(stamp))


class GripperFeedback:
    """Receive-only listener for the gripper frame; never sends anything."""

    def __init__(self, channel):
        self.channel = channel
        self.frame = None
        self.lock = threading.Lock()
        self.error = None
        self.cancel = threading.Event()
        self.bus = self.thread = None

    def start(self):
        import can
        self.bus = can.Bus(interface='socketcan', channel=self.channel,
                           receive_own_messages=False,
                           can_filters=[{'can_id': GRIPPER_FRAME_ID, 'can_mask': 0x7FF, 'extended': False}])
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        try:
            while not self.cancel.is_set():
                msg = self.bus.recv(timeout=.05)
                if msg is None or msg.is_error_frame or msg.is_remote_frame or msg.dlc != 8:
                    continue
                with self.lock:
                    self.frame = (bytes(msg.data), float(msg.timestamp))
        except Exception as error:
            self.error = error

    def latest(self, now=None):
        if self.error is not None:
            raise RuntimeError('夹爪反馈监听线程失败') from self.error
        with self.lock:
            frame = self.frame
        return decode_gripper_frame(frame, time.time() if now is None else now)

    def close(self):
        self.cancel.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.bus is not None:
            self.bus.shutdown()


def wait_for_gripper_feedback(feedback, timeout=3., sleep=time.sleep):
    """Block until the first valid 0x2A8 frame arrives.

    The listener starts (and the effector is initialized) immediately before
    closing, so the first read can beat the first frame: 2026-09-11 13:44
    aborted with '缺少夹爪反馈' 0.00 s after the close request.
    """
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            return feedback.latest()
        except RuntimeError as error:
            last = error
            sleep(.02)
    raise RuntimeError(f'等待夹爪反馈超时（{timeout:g} s）：{last}')


def close_until_blocked(command, read, force_n, step_m=1e-3, step_s=.2,
                        stall_m=.15e-3, stall_s=.7, min_width_m=3e-3,
                        settle_s=.3, timeout_s=45., sleep=time.sleep, event=None):
    """Close in ``step_m`` steps until the jaws stop following the command.

    ``command(width_m)`` sends one gripper target, ``read()`` returns the decoded
    feedback. Returns a dict describing the grip; raises when the jaws closed
    completely without a block, when feedback fails, or on timeout.

    The default pace (1 mm per 200 ms) closes the full ~100 mm travel in about
    20 s; the first run used 0.5 mm per 350 ms and timed out at 83 mm with a
    12 s limit (2026-09-11 13:31).
    """
    if not (0 < force_n <= 50):
        raise ValueError('夹爪力参数必须为 (0,50] N')
    if not (0 < min_width_m < 0.05) or not (0 < step_m <= .01):
        raise ValueError('闭合步长或最小开度无效')
    started = time.monotonic()
    state = read()
    width = state['width_m']
    last_width = width
    progress_at = time.monotonic()
    target = width
    steps = 0
    while True:
        if time.monotonic() - started > timeout_s:
            raise TimeoutError(f'夹爪闭合超时（{timeout_s:g} s，当前开度 {last_width*1000:.1f} mm）')
        # Never ask for more than one step below the width that was measured:
        # the jaws press with the force limit instead of building a large
        # position error inside the block.
        target = max(min_width_m, min(target - step_m, width - step_m))
        command(target)
        steps += 1
        step_end = time.monotonic() + step_s
        while time.monotonic() < step_end:
            state = read()
            sleep(.02)
        state = read()
        width = state['width_m']
        if event is not None:
            event(dict(event='gripper_step', target_mm=target*1000, width_mm=width*1000,
                       force_n=state['force_n'], step=steps))
        if width <= min_width_m:
            raise RuntimeError(f'夹爪已闭合到 {width*1000:.1f} mm，夹爪之间没有物块')
        if width >= last_width - stall_m:
            if time.monotonic() - progress_at >= stall_s:
                break
        else:
            last_width = width
            progress_at = time.monotonic()
    # Hold the last target and require the width to stay put: that is the grip.
    held = read()
    settle_end = time.monotonic() + settle_s
    while time.monotonic() < settle_end:
        sleep(.02)
    settled = read()
    if abs(settled['width_m'] - held['width_m']) > .2e-3:
        raise RuntimeError('夹爪闭合后开度仍在变化，不能确认已夹紧')
    return dict(gripped=True, width_m=settled['width_m'], force_n=settled['force_n'],
                steps=steps, target_mm=target*1000)
