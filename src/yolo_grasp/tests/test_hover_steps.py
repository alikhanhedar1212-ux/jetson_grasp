import importlib.util
from pathlib import Path
import numpy as np
import pytest
from test_arm_console import Arm,Receiver,frames
spec=importlib.util.spec_from_file_location('hover_steps_entry',Path(__file__).resolve().parents[1]/'test_hover_steps.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@pytest.mark.parametrize('angle,stop',[(10000,False),(10200,False),(10400,True)])
def test_guard_stops_excursion(monkeypatch,angle,stop):
    monkeypatch.setattr(m.time,'time',lambda:100.)
    arm=Arm();receiver=Receiver();receiver.frames=frames(100.,joints=(0,angle,-10000,0,0,0))
    c=m.GuardedController(arm,receiver,'/tmp/unused',{},[[-3.14,3.14]]*6)
    c.log=lambda row:None
    c.guard_start=np.deg2rad([0,10,-10,0,0,0]);c.guard_goal=c.guard_start.copy()
    if stop:
        # A single out-of-band sample is no longer enough; the excursion must persist.
        with pytest.raises(RuntimeError):
            for _ in range(m.GUARD_PERSIST_SAMPLES):c.tick()
        assert c.locked and arm.calls==['stop']
    else:
        for _ in range(m.GUARD_PERSIST_SAMPLES):c.tick()
        assert not c.locked and arm.calls==[]

@pytest.mark.parametrize('value',['201','0','-1','nan'])
def test_max_travel_mm_is_validated(monkeypatch,value):
    monkeypatch.setattr(m.sys,'argv',['test_hover_steps.py','--diagnostic','/nonexistent/diagnostic.json',
                                      '--max-travel-mm',value])
    with pytest.raises(ValueError,match=r'\(0,200\]'):
        m.main()

def test_widened_hold_tolerance_ignores_documented_snap(monkeypatch):
    monkeypatch.setattr(m.time,'time',lambda:100.)
    def build():
        arm=Arm();receiver=Receiver();receiver.frames=frames(100.,joints=(0,10700,-10000,0,0,0))
        c=m.GuardedController(arm,receiver,'/tmp/unused',{},[[-3.14,3.14]]*6)
        c.log=lambda row:None
        c.guard_start=np.deg2rad([0,10,-10,0,0,0]);c.guard_goal=c.guard_start.copy()
        return c,arm
    tight,arm=build()
    for _ in range(m.GUARD_PERSIST_SAMPLES-1):tight.tick()
    assert not tight.locked
    with pytest.raises(RuntimeError,match='0.3'):
        tight.tick()
    wide,arm=build()
    wide.guard_tolerance_deg=m.HOLD_TOLERANCE_DEG
    for _ in range(2*m.GUARD_PERSIST_SAMPLES):wide.tick()
    assert not wide.locked and arm.calls==[]

def test_single_sample_spike_is_recorded_not_stopped(monkeypatch):
    monkeypatch.setattr(m.time,'time',lambda:100.)
    arm=Arm();receiver=Receiver();receiver.frames=frames(100.,joints=(0,10400,-10000,0,0,0))
    c=m.GuardedController(arm,receiver,'/tmp/unused',{},[[-3.14,3.14]]*6)
    c.log=lambda row:None
    c.guard_start=np.deg2rad([0,10,-10,0,0,0]);c.guard_goal=c.guard_start.copy()
    seen=[];c.guard_strike_hook=seen.append
    c.tick()
    assert not c.locked and arm.calls==[] and seen[0]['strike']==1
    # In-band samples must clear the streak.
    receiver.frames=frames(100.,joints=(0,10000,-10000,0,0,0))
    c.tick();assert c.guard_strikes==0

def test_hard_limit_stops_on_the_first_sample(monkeypatch):
    monkeypatch.setattr(m.time,'time',lambda:100.)
    arm=Arm();receiver=Receiver();receiver.frames=frames(100.,joints=(0,12000,-10000,0,0,0))
    c=m.GuardedController(arm,receiver,'/tmp/unused',{},[[-3.14,3.14]]*6)
    c.log=lambda row:None
    c.guard_start=np.deg2rad([0,10,-10,0,0,0]);c.guard_goal=c.guard_start.copy()
    with pytest.raises(RuntimeError,match='硬界限'):
        c.tick()
    assert c.locked

def test_zero_zone_reading_is_not_treated_as_movement(monkeypatch):
    """0.000 deg inside the zero zone: the motor frame showed -0.286 instead."""
    monkeypatch.setattr(m.time,'time',lambda:100.)
    arm=Arm();receiver=Receiver();receiver.frames=frames(100.,joints=(0,0,-10000,0,0,0))
    c=m.GuardedController(arm,receiver,'/tmp/unused',{},[[-3.14,3.14]]*6)
    c.log=lambda row:None
    c.guard_start=np.deg2rad([0,-0.4,-10,0,0,0]);c.guard_goal=c.guard_start.copy()
    for _ in range(10):c.tick()
    assert not c.locked and arm.calls==[] and c.guard_strikes==0

def test_count_level_excursion_is_noise(monkeypatch):
    """Five counts (0.005 deg) past the band is quantisation, not movement."""
    monkeypatch.setattr(m.time,'time',lambda:100.)
    arm=Arm();receiver=Receiver();receiver.frames=frames(100.,joints=(0,10305,-10000,0,0,0))
    c=m.GuardedController(arm,receiver,'/tmp/unused',{},[[-3.14,3.14]]*6)
    c.log=lambda row:None
    c.guard_start=np.deg2rad([0,10,-10,0,0,0]);c.guard_goal=c.guard_start.copy()
    for _ in range(10):c.tick()
    assert not c.locked and c.guard_strikes==0
