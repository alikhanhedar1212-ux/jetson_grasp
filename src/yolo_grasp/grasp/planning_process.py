"""Persistent pure-planning process. Requests contain numbers, never robot handles."""
import os
from queue import Empty
from types import SimpleNamespace

from .worker_ipc import (context, single_thread_environment, configure_worker,
                         LatestSlot, publish_until_delivered, stop_process, wait_for_result,
                         WorkerRejected)


def compute_plan(request):
    import numpy as np
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh, get_mdh
    from yolo_grasp.hover_red_block import make_path, replan_from_current
    from .geometry import pose_matrix, transform
    from .hover_test import diagnose, diagnose_best_tool_yaw
    from .hover_motion import timed_path
    mdh = get_mdh('piper')
    fk = lambda q: pose_matrix(fk_from_mdh(mdh, list(q)))
    limits, max_delta = request['limits'], request['max_delta']
    kind = request['kind']
    if kind == 'initial':
        diagnose_fn = diagnose_best_tool_yaw if request.get('tool_yaw_search', False) else diagnose
        report = diagnose_fn(request['joints'], transform(request['flange']), request['camera_point'],
                          request['handeye'], limits, fk, request['clearance_mm']/1000,
                          max_delta, request['max_travel_mm'],
                          np.asarray(request['target_offset_mm'])/1000,
                          np.asarray(request['tcp_offset_mm'])/1000)
    elif kind == 'replan':
        args = SimpleNamespace(clearance_mm=request['clearance_mm'], max_joint_delta_deg=max_delta,
                               max_travel_mm=request['max_travel_mm'],
                               target_offset_mm=np.asarray(request['target_offset_mm']),
                               tcp_offset_mm=np.asarray(request['tcp_offset_mm']),
                               tool_yaw_search=request.get('tool_yaw_search', False))
        report, trajectory = replan_from_current(
            request['base_point'], SimpleNamespace(joints=np.asarray(request['joints'])),
            np.asarray(request['flange']), np.asarray(request['origin_flange']),
            np.asarray(request['handeye']), limits, fk, args)
        return dict(report=report, trajectory=trajectory)
    elif kind == 'path':
        report = request['report']
    else:
        raise ValueError(f'Unknown planning request: {kind}')
    try:
        path = np.asarray(make_path(report, limits, fk, max_delta))
    except ValueError as error:
        error.diagnostic = report
        raise
    return dict(report=report, path=path,
                trajectory=timed_path(path, speed_deg_s=request.get('speed_deg_s', 2.)))


def planning_worker(requests, results, stop, compute=compute_plan):
    configure_worker(1)
    publish_until_delivered(results, dict(request_id=0, result=dict(pid=os.getpid())), stop)
    while not stop.is_set():
        try:
            request_id, request = requests.get(timeout=.05)
        except Empty:
            continue
        try:
            result = dict(request_id=request_id, result=compute(request))
        except Exception as error:
            result = dict(request_id=request_id, error=str(error), error_type=type(error).__name__,
                          diagnostic=getattr(error, 'diagnostic', None))
        publish_until_delivered(results, result, stop)


class PlanningProcess:
    def __init__(self, compute=compute_plan):
        ctx = context()
        self.requests, self.results, self.stop = ctx.Queue(maxsize=1), LatestSlot(ctx), ctx.Event()
        self.process = ctx.Process(target=planning_worker,
                                   args=(self.requests, self.results, self.stop, compute), daemon=True)
        self.request_id = 0
        self.busy = False

    def start(self, tick=lambda: None):
        with single_thread_environment():
            self.process.start()
        try:
            return wait_for_result(self.process, self.results, 0, tick, self.stop, timeout=30.)
        except BaseException:
            self.close()
            raise

    def run(self, request, tick, *, timeout=60.):
        if self.busy or self.stop.is_set() or not self.process.is_alive():
            raise RuntimeError('规划进程未就绪或已有任务')
        self.busy = True
        self.request_id += 1
        try:
            self.requests.put_nowait((self.request_id, request))
            return wait_for_result(self.process, self.results, self.request_id, tick,
                                   self.stop, timeout=timeout)
        except WorkerRejected:
            raise  # A rejected path is recoverable; the worker can accept a retry.
        except BaseException:
            self.close()  # Never leave an abandoned CPU job running after STOP/timeout.
            raise
        finally:
            self.busy = False

    def close(self):
        stop_process(self.process, self.stop)
        self.requests.cancel_join_thread()
        self.requests.close()
