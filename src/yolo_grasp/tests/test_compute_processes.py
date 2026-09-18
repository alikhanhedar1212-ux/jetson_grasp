import os
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from grasp.planning_process import PlanningProcess
from grasp.vision_process import ProcessPreview
from grasp.worker_ipc import context, LatestSlot, publish_until_delivered
from grasp.hover_observation import target_point


def busy_compute(request):
    if request.get('reject'):
        raise ValueError('unreachable target')
    if request.get('crash'):
        os._exit(7)
    deadline = time.monotonic()+request.get('seconds', .1)
    value = 1
    while time.monotonic() < deadline:
        value = (value*17+3) % 1000003  # GIL-bound work, no sleeping/native acceleration.
    return dict(pid=os.getpid(), tag=request.get('tag'), value=value)


def fake_vision(cfg, model_path, imgsz, device, latest, saves, saved, stop, max_fps):
    from queue import Empty
    counter = 0
    while not stop.is_set():
        stamp = time.monotonic()
        latest.publish(dict(stamp=stamp, metadata=dict(frame_number=counter,
                            color_timestamp_ms=time.time()*1000,
                            timestamp_domain='timestamp_domain.global_time'),
                            point=dict(valid=True, xyz_camera_m=[0., 0., .2]),
                            annotated=np.full((4, 4, 3), counter % 255, dtype=np.uint8),
                            names={0: 'red_block'}, timings=(.001, .002, stamp), pid=os.getpid()))
        counter += 1
        try:
            request_id, requested_stamp, directory = saves.get_nowait()
        except Empty:
            pass
        else:
            publish_until_delivered(saved, dict(request_id=request_id, result=requested_stamp), stop)
        stop.wait(.01)


def crashing_vision(*args):
    os._exit(9)


class CV:
    WINDOW_NORMAL = FONT_HERSHEY_SIMPLEX = WND_PROP_VISIBLE = 0
    def destroyAllWindows(self): pass


def wait_sample(preview):
    deadline = time.monotonic()+5
    while preview.sample() is None:
        if time.monotonic() > deadline:
            pytest.fail('worker did not publish')
        time.sleep(.01)
    return preview.sample()


def test_latest_slot_overwrites_backlog_and_reader_never_waits():
    slot = LatestSlot(context(), 1024)
    for index in range(100):
        assert slot.publish(dict(index=index))
    version, packet = slot.read()
    assert packet == dict(index=99)
    assert slot.read(version) == (version, None)
    slot.lock.acquire()
    try:
        start = time.monotonic()
        assert slot.read(version) == (version, None)
        assert not slot.publish(dict(index=100))
        assert time.monotonic()-start < .05
    finally:
        slot.lock.release()
    with pytest.raises(ValueError, match='capacity'):
        slot.publish(bytes(2048))


def test_planner_process_leaves_parent_ticks_running_and_supports_retry():
    planner = PlanningProcess(busy_compute)
    try:
        info = planner.start()
        stamps = []
        result = planner.run(dict(seconds=.4, tag='first'), lambda: stamps.append(time.monotonic()))
        assert result['pid'] == info['pid'] != os.getpid()
        assert result['tag'] == 'first'
        assert len(stamps) >= 10
        assert max(np.diff(stamps)) < .25
        with pytest.raises(ValueError, match='unreachable'):
            planner.run(dict(reject=True), lambda: None)
        assert planner.run(dict(tag='second'), lambda: None)['tag'] == 'second'
    finally:
        planner.close()
    assert not planner.process.is_alive()


@pytest.mark.parametrize('fault', ['timeout', 'stop', 'control_error', 'crash'])
def test_planner_failure_terminates_job_and_cannot_resume_it(fault):
    planner = PlanningProcess(busy_compute)
    try:
        planner.start()
        def tick():
            if fault == 'stop': raise KeyboardInterrupt('operator stop')
            if fault == 'control_error': raise ValueError('CAN control failure')
        error = {'timeout': TimeoutError, 'stop': KeyboardInterrupt,
                 'control_error': ValueError, 'crash': RuntimeError}[fault]
        with pytest.raises(error):
            planner.run(dict(seconds=10., crash=fault == 'crash'), tick,
                        timeout=.1 if fault == 'timeout' else 5.)
        assert not planner.process.is_alive()
        with pytest.raises(RuntimeError, match='未就绪'):
            planner.run({}, lambda: None)
    finally:
        planner.close()


def test_vision_process_has_compact_fresh_samples_and_archives_exact_stamp(tmp_path):
    preview = ProcessPreview({}, 'unused', CV(), show_window=False, worker=fake_vision)
    try:
        preview.start()
        sample = wait_sample(preview)
        assert preview.process.pid != os.getpid()
        frame = sample[1]
        assert not hasattr(frame, 'points') and not hasattr(frame, 'depth_m')
        assert sample[2] is None  # No torch/model object in the controller.
        assert target_point(sample, {}, None)['xyz_camera_m'] == [0., 0., .2]
        assert preview.save_sample(sample, tmp_path, lambda: None) == sample[0]
        first = frame.metadata['frame_number']
        time.sleep(.1)
        assert preview.sample()[1].metadata['frame_number'] > first
        preview.pump()
    finally:
        preview.close()
    assert not preview.process.is_alive()


def test_vision_crash_is_reported_instead_of_reusing_stale_frame():
    preview = ProcessPreview({}, 'unused', CV(), show_window=False, worker=crashing_vision)
    try:
        preview.start()
        preview.process.join(5)
        with pytest.raises(RuntimeError, match='已退出'):
            preview.sample()
    finally:
        preview.close()


def test_can_backlog_drained_with_original_timestamps(monkeypatch):
    from grasp.robot import CanFeedback
    receiver = CanFeedback.__new__(CanFeedback)
    receiver.cancel, receiver.lock = threading.Event(), threading.Lock()
    receiver.frames, receiver.error = {}, None
    receiver.receive_stats = dict(messages=0, last_batch=0, last_lag_ms=0., max_lag_ms=0.)
    frames = [SimpleNamespace(arbitration_id=0x2a5+(i % 3), data=bytes([i])*8,
                              timestamp=10.+i*.001, is_error_frame=False,
                              is_remote_frame=False, dlc=8) for i in range(30)]
    timeouts = []
    def receive(timeout):
        timeouts.append(timeout)
        if frames: return frames.pop(0)
        receiver.cancel.set()
        return None
    receiver.bus = SimpleNamespace(recv=receive)
    monkeypatch.setattr('grasp.robot.time.time', lambda: 10.5)
    receiver._read()
    assert receiver.error is None
    assert timeouts[0] == .05 and all(t == 0 for t in timeouts[1:])
    for index in range(27, 30):
        assert receiver.snapshot()[0x2a5+(index % 3)] == (bytes([index])*8, 10.+index*.001)
    assert receiver.diagnostics()['last_lag_ms'] > 250  # Never restamp old frames as fresh.


def test_real_initial_replan_and_lift_paths_in_spawned_worker():
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from grasp.geometry import pose_matrix, matrix_pose, pose_error
    from grasp.mat_pose import TARGET_DEG
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    joints = np.deg2rad(TARGET_DEG)
    flange = fk(joints)
    wanted = flange.copy(); wanted[:3, 3] += [.03, -.02, -.03]
    base = wanted[:3, 3]+wanted[:3, :3] @ [0., 0., .08]-[0., 0., .035]
    camera = (np.linalg.inv(flange) @ np.r_[base, 1])[:3]
    common = dict(limits=list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values()),
                  max_delta=90., clearance_mm=35., max_travel_mm=200.,
                  handeye=np.eye(4), target_offset_mm=[0., 0., 0.], tcp_offset_mm=[0., 0., 80.])
    planner = PlanningProcess()
    try:
        planner.start()
        result = planner.run(dict(common, kind='initial', joints=joints,
                                   flange=flange.tolist(), camera_point=camera), lambda: None)
        assert not result['report']['issues']
        targets = result['trajectory'][1]
        assert pose_error(fk(targets[-1]), wanted)[0] < .001
        stopped_joints = targets[len(targets)//2]
        stopped = fk(stopped_joints)
        moved_base = base+[.01, .005, 0.]
        result = planner.run(dict(common, kind='replan', joints=stopped_joints,
                                   flange=stopped, origin_flange=flange, base_point=moved_base), lambda: None)
        np.testing.assert_allclose(result['report']['current_joints_rad'], stopped_joints)
        np.testing.assert_allclose(result['report']['red_block_base_xyz_m'], moved_base)
        goal = wanted.copy(); goal[:3, 3] += [.01, .005, 0.]
        end_joints = result['trajectory'][1][-1]
        assert pose_error(fk(end_joints), goal)[0] < .001
        lifted = goal.copy(); lifted[2, 3] += .02
        report = dict(issues=[], current_joints_rad=end_joints.tolist(),
                      current_CAN_flange_m_rad=matrix_pose(fk(end_joints)),
                      hypothetical_target_flange_m_rad=matrix_pose(lifted))
        result = planner.run(dict(common, kind='path', report=report, speed_deg_s=8.), lambda: None)
        assert pose_error(fk(result['trajectory'][1][-1]), lifted)[0] < .001
    finally:
        planner.close()
