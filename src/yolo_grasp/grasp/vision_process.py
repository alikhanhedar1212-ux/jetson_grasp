"""Continuous RGB-D/YOLO worker with compact latest-result IPC and on-demand archives."""
from collections import OrderedDict
import os
from pathlib import Path
from queue import Empty
import time
from types import SimpleNamespace

from .hover_preview import HoverPreview
from .worker_ipc import (context, single_thread_environment, configure_worker, LatestSlot,
                         publish_until_delivered, stop_process, wait_for_result)


def vision_worker(cfg, model_path, imgsz, device, latest, saves, saved, stop, max_fps):
    try:
        configure_worker(0)
        publish_until_delivered(latest, dict(status='Loading YOLO in vision process'), stop)
        import cv2
        import torch
        from ultralytics import YOLO
        from .camera import RealSense
        from .hover_observation import target_point
        from .live_depth import locate_top_surface
        cv2.setNumThreads(1)
        torch.set_num_threads(1)
        model = YOLO(model_path)
        cache = OrderedDict()
        previous_box_xyxy = None
        with RealSense(cfg) as camera:
            publish_until_delivered(latest, dict(status='Warming up YOLO in vision process'), stop)
            warmup = camera.capture(discard=0)
            model.predict(source=warmup.bgr, device=device, imgsz=imgsz, conf=.6, verbose=False)
            # Backend initialization may have reset torch's thread count.
            torch.set_num_threads(1)
            del warmup
            while not stop.is_set():
                try:
                    request_id, stamp, directory = saves.get_nowait()
                except Empty:
                    pass
                else:
                    try:
                        if stamp not in cache:
                            raise ValueError('待保存观测已过期，拒绝替换成其它帧')
                        frame, annotated = cache[stamp]
                        directory = Path(directory)
                        frame.save(directory/'observation')
                        if not cv2.imwrite(str(directory/'detection.png'), annotated):
                            raise OSError('无法保存检测画面')
                        reply = dict(request_id=request_id, result=True)
                    except Exception as error:
                        reply = dict(request_id=request_id, error=str(error),
                                     error_type=type(error).__name__)
                    publish_until_delivered(saved, reply, stop)
                started = time.monotonic()
                frame = camera.capture(discard=0)
                captured = time.monotonic()
                prediction = model.predict(source=frame.bgr, device=device, imgsz=imgsz,
                                           conf=.6, verbose=False)[0]
                annotated = prediction.plot()
                point = target_point((started, frame, prediction, annotated), model.names,
                                     locate_top_surface,
                                     previous_box_xyxy=previous_box_xyxy)
                if point.get('valid') and point.get('selected_box_xyxy') is not None:
                    previous_box_xyxy = point['selected_box_xyxy']
                completed = time.monotonic()
                cache[started] = (frame, annotated)
                while len(cache) > 8:
                    cache.popitem(last=False)
                # No torch tensors, depth image or point cloud cross into the controller.
                latest.publish(dict(stamp=started, metadata=frame.metadata, point=point,
                                    annotated=annotated, names=model.names, pid=os.getpid(),
                                    timings=(captured-started, completed-captured, completed)))
                stop.wait(max(0., 1/max_fps-(time.monotonic()-started)))
    except BaseException as error:
        publish_until_delivered(latest, dict(error=f'{type(error).__name__}: {error}'), stop)


class ProcessPreview(HoverPreview):
    """GUI stays on the main thread; camera, depth and model exist only in the child."""
    def __init__(self, cfg, model_path, cv2, imgsz=320, device='cpu', show_window=True,
                 *, worker=vision_worker):
        super().__init__(None, None, cv2, imgsz, device, show_window)
        ctx = context()
        self.slot, self.saved = LatestSlot(ctx), LatestSlot(ctx, 16384)
        self.saves, self.stop_event = ctx.Queue(maxsize=1), ctx.Event()
        self.process = ctx.Process(target=worker, args=(
            cfg, str(model_path), imgsz, device, self.slot, self.saves, self.saved,
            self.stop_event, 10.), daemon=True)
        self.version = self.save_id = 0
        self.names = {}
        self.display_interval = .1
        self.startup_timeout = 60.

    def start(self):
        self.started = time.monotonic()
        if self.show_window:
            self.cv2.namedWindow(self.title, self.cv2.WINDOW_NORMAL)
        with single_thread_environment():
            self.process.start()

    def sample(self):
        self.version, packet = self.slot.read(self.version)
        if packet is not None:
            if 'error' in packet:
                self.error = RuntimeError(packet['error'])
            elif 'stamp' in packet:
                frame = SimpleNamespace(metadata=packet['metadata'], detected_point=packet['point'])
                self.latest = (packet['stamp'], frame, None, packet['annotated'])
                self.timings, self.names = packet['timings'], packet['names']
            elif self.latest is None:
                self.stage = packet['status']
        if self.error is not None:
            raise RuntimeError(f'视觉子进程失败：{self.error}') from self.error
        if self.process.pid is not None and not self.process.is_alive() and not self.stop_event.is_set():
            raise RuntimeError(f'视觉子进程已退出，exitcode={self.process.exitcode}')
        return self.latest

    def pause_inference(self, sample=None):
        raise RuntimeError('抓取视觉子进程保持持续检测，不支持暂停推理')

    def save_sample(self, sample, directory, tick):
        self.save_id += 1
        self.saves.put_nowait((self.save_id, sample[0], str(directory)))
        return wait_for_result(self.process, self.saved, self.save_id, tick,
                               self.stop_event, timeout=10.)

    def close(self):
        stop_process(self.process, self.stop_event)
        self.saves.cancel_join_thread()
        self.saves.close()
        if self.show_window:
            self.cv2.destroyAllWindows()
