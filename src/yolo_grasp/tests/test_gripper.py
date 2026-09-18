import struct
import time
import pytest
from grasp import gripper as m


def frame(width_m, force_n=0., flags=0, mode=0, stamp=None):
    return (struct.pack('>ihBB', int(round(width_m*1e6)), int(round(force_n*1e3)), flags, mode),
            time.time() if stamp is None else stamp)


def test_decode_rejects_missing_stale_and_faulted_frames():
    with pytest.raises(RuntimeError, match='缺少'): m.decode_gripper_frame(None, 100.)
    now = 100.
    with pytest.raises(RuntimeError, match='过期'): m.decode_gripper_frame(frame(.02, stamp=90.), now)
    with pytest.raises(RuntimeError, match='长度'): m.decode_gripper_frame((b'\x00'*7, now), now)
    with pytest.raises(RuntimeError, match='故障'): m.decode_gripper_frame(frame(.02, flags=0x02, stamp=now), now)
    with pytest.raises(RuntimeError, match='非宽度模式'): m.decode_gripper_frame(frame(.02, mode=1, stamp=now), now)


def test_decode_returns_width_force_and_active_flag():
    now = 100.
    state = m.decode_gripper_frame(frame(.025, 3.5, flags=0x40, stamp=now), now)
    assert state['width_m'] == pytest.approx(.025) and state['force_n'] == pytest.approx(3.5)
    assert state['active'] is True
    state = m.decode_gripper_frame(frame(.0, 0., stamp=now - .1), now)
    assert state['width_m'] == pytest.approx(0.)


def test_decode_rejects_an_absurd_width_so_the_wait_keeps_looking():
    """2026-09-14: the first "ready" frame decoded 16777.5 mm and the close aborted."""
    now = 100.
    with pytest.raises(RuntimeError, match='开度异常'):
        m.decode_gripper_frame(frame(16.7775, stamp=now), now)
    # 100 mm 满行程仍然合法
    assert m.decode_gripper_frame(frame(.1, stamp=now), now)['width_m'] == pytest.approx(.1)


class FakeGripper:
    """Width follows commands until it reaches ``block_m`` (then it stalls)."""
    def __init__(self, start_m=.06, block_m=None, follow_gain=1.0, freeze=False):
        self.width = start_m
        self.block_m = block_m
        self.commanded = []
        self.gain = follow_gain
        self.freeze = freeze
    def command(self, width_m):
        self.commanded.append(width_m)
        if self.freeze:
            return
        wanted = max(width_m, self.block_m if self.block_m is not None else 0.)
        self.width += (wanted - self.width) * self.gain
    def read(self):
        return dict(width_m=self.width, force_n=1.0, active=True, stamp=time.time())


def test_closing_stops_when_the_block_blocks_the_jaws():
    grip = FakeGripper(start_m=.06, block_m=.025)
    result = m.close_until_blocked(grip.command, grip.read, force_n=3.,
                                   step_m=.5e-3, sleep=lambda s: None, step_s=0.,
                                   stall_s=0., settle_s=0.)
    assert result['gripped'] and result['width_m'] == pytest.approx(.025, abs=1e-3)
    assert min(grip.commanded) >= .025 - 1e-6 - .5e-3


def test_closing_without_a_block_is_refused():
    grip = FakeGripper(start_m=.06, block_m=None)
    with pytest.raises(RuntimeError, match='没有物块'):
        m.close_until_blocked(grip.command, grip.read, force_n=3.,
                              sleep=lambda s: None, step_s=0., stall_s=0., settle_s=0.)


def test_unmoving_jaws_are_reported_as_a_grip_not_a_fault():
    grip = FakeGripper(start_m=.02, freeze=True)
    result = m.close_until_blocked(grip.command, grip.read, force_n=3.,
                                   sleep=lambda s: None, step_s=0., stall_s=0., settle_s=0.)
    assert result['gripped'] and result['width_m'] == pytest.approx(.02)


def test_stale_feedback_aborts_the_close():
    def read(): raise RuntimeError('夹爪反馈过期或时钟异常（+2.000 s）')
    with pytest.raises(RuntimeError, match='过期'):
        m.close_until_blocked(lambda w: None, read, force_n=3.,
                              sleep=lambda s: None, step_s=0., stall_s=0., settle_s=0.)


def test_waiting_for_the_first_frame_retries_then_times_out():
    calls = {'n': 0}
    class Late:
        def latest(self):
            calls['n'] += 1
            if calls['n'] < 4:
                raise RuntimeError('缺少夹爪反馈')
            return dict(width_m=.06, force_n=0., active=False, stamp=0.)
    assert m.wait_for_gripper_feedback(Late(), timeout=1., sleep=lambda s: None)['width_m'] == .06

    class Missing:
        def latest(self): raise RuntimeError('缺少夹爪反馈')
    with pytest.raises(RuntimeError, match='等待夹爪反馈超时'):
        m.wait_for_gripper_feedback(Missing(), timeout=.01, sleep=lambda s: None)


def test_close_validates_force_and_step():
    grip = FakeGripper(block_m=.025)
    with pytest.raises(ValueError, match='力参数'):
        m.close_until_blocked(grip.command, grip.read, force_n=0.)
    with pytest.raises(ValueError, match='步长'):
        m.close_until_blocked(grip.command, grip.read, force_n=3., step_m=0.)
