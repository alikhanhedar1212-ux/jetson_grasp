from types import SimpleNamespace
import pytest
from grasp import reset_estop as recovery
from test_arm_console import frames


@pytest.mark.parametrize('case', ['success', 'normal', 'enabled', 'fault', 'drift', 'stale', 'timeout', 'not_arrived'])
def test_recovery_is_one_shot_and_never_enables_or_moves(monkeypatch, case):
    clock = [100.]
    resets = []
    monkeypatch.setattr(recovery.time, 'time', lambda: clock[0])
    monkeypatch.setattr(recovery.time, 'monotonic', lambda: clock[0])
    monkeypatch.setattr(recovery.time, 'sleep', lambda dt: clock.__setitem__(0, clock[0] + dt))

    def snapshot():
        stamp = 99 if case == 'stale' else clock[0]
        joints = (0, 10000, -10000, 0, 0, 500 if case == 'drift' and clock[0] > 100.1 else 0)
        result = frames(stamp, enabled=case == 'enabled', joints=joints)
        status = 0 if case == 'normal' or (case in ('success', 'not_arrived') and resets) else 1
        result[0x2A1] = (bytes([0, status, 1, 0, 1 if case == 'not_arrived' else 0, 0, 0, 1 if case == 'fault' else 0]), stamp)
        return result

    # Deliberately no enable/move methods: recovery must not call either.
    arm = SimpleNamespace(reset=lambda: resets.append(clock[0]))
    receiver = SimpleNamespace(snapshot=snapshot)
    if case in ('success', 'normal', 'not_arrived'):
        recovery.recover(arm, receiver, emit=lambda _: None)
    else:
        with pytest.raises(RuntimeError):
            recovery.recover(arm, receiver, emit=lambda _: None)
    assert len(resets) == (1 if case in ('success', 'timeout', 'not_arrived') else 0)
