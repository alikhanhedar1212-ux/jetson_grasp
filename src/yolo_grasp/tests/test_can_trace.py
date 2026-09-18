import sys
import time
import types
from types import SimpleNamespace
import pytest
from grasp.can_trace import CanTrace, TRACE_IDS


def install_fake_can(monkeypatch):
    created = {}
    class Bus:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.frames = []
            self.shutdown_called = False
            created['bus'] = self
        def recv(self, timeout=None):
            if self.frames:
                return self.frames.pop(0)
            time.sleep(.005)
            return None
        def shutdown(self):
            self.shutdown_called = True
    module = types.ModuleType('can')
    module.Bus = Bus
    monkeypatch.setitem(sys.modules, 'can', module)
    return created


def frame(can_id=0x2A5, data=bytes(range(8))):
    return SimpleNamespace(is_error_frame=False, is_remote_frame=False, dlc=8,
                           arbitration_id=can_id, data=data, timestamp=1.0)


def test_trace_is_receive_only_and_covers_both_channels(monkeypatch):
    created = install_fake_can(monkeypatch)
    trace = CanTrace('can1', window_s=30.).start()
    bus = created['bus']
    assert bus.kwargs['receive_own_messages'] is False
    assert sorted(f['can_id'] for f in bus.kwargs['can_filters']) == sorted(TRACE_IDS)
    assert set(TRACE_IDS) >= {0x2A5, 0x2A7, 0x251, 0x256}
    bus.frames.append(frame(0x251, bytes([0, 1, 0, 2, 0, 0, 0, 3])))
    time.sleep(.05)
    rows = trace.dump()
    assert len(rows) == 1 and rows[0]['can_id'] == '0x251'
    assert rows[0]['data_hex'] == '0001000200000003'
    trace.close()
    assert bus.shutdown_called


def test_trace_drops_frames_older_than_the_window(monkeypatch):
    created = install_fake_can(monkeypatch)
    trace = CanTrace('can1', window_s=.02).start()
    created['bus'].frames.append(frame(0x2A6))
    time.sleep(.05)
    created['bus'].frames.append(frame(0x2A7))
    time.sleep(.05)
    assert [row['can_id'] for row in trace.dump()] == ['0x2a7']
    trace.close()


def test_dead_listener_is_reported_instead_of_returning_stale_data(monkeypatch):
    created = install_fake_can(monkeypatch)
    trace = CanTrace('can1').start()
    created['bus'].recv = lambda timeout=None: (_ for _ in ()).throw(RuntimeError('bus off'))
    time.sleep(.05)
    with pytest.raises(RuntimeError, match='记录线程失败'):
        trace.dump()
    trace.close()


def test_ignored_frame_shapes_are_skipped(monkeypatch):
    created = install_fake_can(monkeypatch)
    trace = CanTrace('can1', window_s=30.).start()
    bad = frame(0x2A5); bad.dlc = 4
    error = frame(0x2A5); error.is_error_frame = True
    created['bus'].frames.extend([bad, error, frame(0x2A5)])
    time.sleep(.05)
    assert len(trace.dump()) == 1
    trace.close()
