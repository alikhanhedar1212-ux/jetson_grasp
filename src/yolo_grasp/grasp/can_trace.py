"""Receive-only dual-channel CAN trace with a bounded rolling buffer.

The joint-state frames 0x2A5-0x2A7 and the per-motor frames 0x251-0x256 are two
independent reports of the same six joints. Recording both lets a single-sample
joint-frame outlier be classified as a real movement or as a feedback artefact
(see BASE_AXES_TEST.md). This listener only receives: it never sends a frame and
it is not on the control path, so a failure here cannot change the motion.
"""
import threading
import time
from collections import deque


JOINT_FRAME_IDS = (0x2A5, 0x2A6, 0x2A7)
MOTOR_FRAME_IDS = (0x251, 0x252, 0x253, 0x254, 0x255, 0x256)
TRACE_IDS = JOINT_FRAME_IDS + MOTOR_FRAME_IDS


class CanTrace:
    """Passive SocketCAN listener keeping the last ``window_s`` of frames."""

    def __init__(self, channel, window_s=2.0, max_rows=20000):
        self.channel = channel
        self.window_s = window_s
        self.rows = deque(maxlen=max_rows)
        self.lock = threading.Lock()
        self.error = None
        self.cancel = threading.Event()
        self.bus = self.thread = None

    def start(self):
        import can
        self.bus = can.Bus(interface="socketcan", channel=self.channel,
                           receive_own_messages=False,
                           can_filters=[{"can_id": i, "can_mask": 0x7FF, "extended": False}
                                        for i in TRACE_IDS])
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()
        return self

    def _read(self):
        try:
            while not self.cancel.is_set():
                msg = self.bus.recv(timeout=.05)
                if (msg is None or msg.is_error_frame or msg.is_remote_frame or msg.dlc != 8):
                    continue
                now = time.time()
                row = dict(host_s=now, can_timestamp_s=float(msg.timestamp),
                           can_id=hex(msg.arbitration_id), data_hex=bytes(msg.data).hex())
                with self.lock:
                    self.rows.append(row)
                    while self.rows and now - self.rows[0]["host_s"] > self.window_s:
                        self.rows.popleft()
        except Exception as error:      # a dead listener must not look like data
            self.error = error

    def dump(self):
        """Frames currently buffered, oldest first; raises if the listener died."""
        if self.error is not None:
            raise RuntimeError("双通道 CAN 记录线程失败") from self.error
        with self.lock:
            return list(self.rows)

    def close(self):
        self.cancel.set()
        if self.thread is not None:
            self.thread.join(timeout=1)
        if self.bus is not None:
            self.bus.shutdown()
