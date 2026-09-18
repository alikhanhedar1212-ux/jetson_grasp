"""Bounded, nonblocking latest-value IPC for compute-only spawned workers."""
from contextlib import contextmanager
import ctypes
import multiprocessing as mp
import os
import pickle
import time


class WorkerRejected(ValueError):
    """A completed request was rejected; distinct from a control-loop failure."""


def context():
    # Never fork a process holding CAN sockets, CUDA state or Python locks.
    return mp.get_context('spawn')


@contextmanager
def single_thread_environment():
    keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
    previous = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = '1'
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def configure_worker(index):
    """Keep compute workers away from the first two allowed CPUs when possible."""
    if hasattr(os, 'sched_getaffinity'):
        cpus = sorted(os.sched_getaffinity(0))
        if len(cpus) >= 4:
            os.sched_setaffinity(0, {cpus[2:][index % (len(cpus)-2)]})
    if hasattr(os, 'nice'):
        os.nice(5)


class LatestSlot:
    """One fixed-size slot, never a frame queue; consumers never wait on its lock."""
    def __init__(self, ctx, capacity=4*1024*1024):
        self.buffer = ctx.RawArray(ctypes.c_ubyte, capacity)
        self.size = ctx.RawValue(ctypes.c_size_t, 0)
        self.version = ctx.RawValue(ctypes.c_uint64, 0)
        self.lock = ctx.Lock()

    def publish(self, value):
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if len(payload) > len(self.buffer):
            raise ValueError('Worker result exceeds fixed IPC capacity')
        if not self.lock.acquire(False):
            return False
        try:
            memoryview(self.buffer).cast('B')[:len(payload)] = payload
            self.size.value = len(payload)
            self.version.value += 1
        finally:
            self.lock.release()
        return True

    def read(self, previous=0):
        if not self.lock.acquire(False):
            return previous, None
        try:
            version = self.version.value
            if version == previous or not self.size.value:
                return previous, None
            payload = bytes(memoryview(self.buffer).cast('B')[:self.size.value])
        finally:
            self.lock.release()
        return version, pickle.loads(payload)


def publish_until_delivered(slot, value, stop):
    while not stop.is_set():
        if slot.publish(value):
            return
        stop.wait(.005)


def stop_process(process, stop):
    stop.set()
    if process is None or process.pid is None:
        return
    process.join(timeout=.3)
    if process.is_alive():
        process.terminate()
        process.join(timeout=.5)
    if process.is_alive():
        process.kill()
        process.join(timeout=.5)
    if process.is_alive():
        raise RuntimeError('计算子进程未能退出')


def wait_for_result(process, slot, request_id, tick, stop, *, timeout=60.):
    deadline = time.monotonic()+timeout
    version = 0
    while True:
        tick()
        version, packet = slot.read(version)
        if packet is not None and packet.get('request_id') == request_id:
            if packet.get('error'):
                kind = WorkerRejected if packet.get('error_type') == 'ValueError' else RuntimeError
                error = kind(packet['error'])
                error.diagnostic = packet.get('diagnostic')
                raise error
            return packet['result']
        if not process.is_alive():
            raise RuntimeError(f'计算子进程异常退出，exitcode={process.exitcode}')
        if time.monotonic() >= deadline:
            raise TimeoutError(f'计算子进程等待结果超过{timeout:g}秒')
        stop.wait(.01)
