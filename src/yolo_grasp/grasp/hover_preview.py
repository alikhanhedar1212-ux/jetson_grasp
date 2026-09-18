"""Background RGB-D inference; GUI and stop keys stay on the control thread."""
import threading
import time

import numpy as np


class HoverPreview:
    title = 'D405 red block | terminal: go | Q / Esc / Space: STOP'

    def __init__(self, camera_factory, model, cv2, imgsz=640, device='0', show_window=True):
        self.camera_factory = camera_factory
        self.model = model
        self.cv2 = cv2
        # Inference size dominates this pipeline's memory use on the Jetson;
        # 640 OOMs on a busy 8 GB board, 416/320 are the fallbacks.
        self.imgsz = int(imgsz)
        # 'cpu' avoids the CUDA/cuBLAS context entirely, which is what fails
        # first when the Jetson's unified memory is tight.
        self.device = device
        self.inference_threads = None
        # Headless mode still captures and infers; it only skips the window.
        self.show_window = bool(show_window)
        # Inference is the CPU hog that starves the CAN reader thread; once the
        # target is locked, only lightweight colour capture continues for display.
        self.paused = False
        self.pause_ack = threading.Event()
        self.locked_sample = None
        self.live_color = None
        self.paused_at = None
        # Cap for the unpaused detection rate. The placement leg keeps detecting
        # while a pre-planned path runs; running flat out there starves the CAN
        # reader thread, which is what the pause was originally protecting.
        self.min_interval = 0.
        self.stage = 'Starting camera'
        self.detail = ''
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.latest = None
        self.error = None
        self.timings = None
        self.started = time.monotonic()
        self.thread = None
        self.display_interval = 0.
        self.last_draw = float('-inf')
        self.startup_timeout = 30.

    def start(self):
        self.started = time.monotonic()
        if self.show_window:
            self.cv2.namedWindow(self.title, self.cv2.WINDOW_NORMAL)
        self.thread = threading.Thread(target=self._capture, daemon=True)
        self.thread.start()

    def release_cuda_cache(self):
        """Hand the warm-up allocations back; the Jetson pool is shared with the OS."""
        if str(self.device).lower() == 'cpu':
            return   # never touch CUDA when the caller asked for CPU inference
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except BaseException as error:
            print(f'CUDA 缓存回收跳过：{error}', flush=True)

    def pause_inference(self, sample=None):
        """Request a pause and freeze the exact observation used for planning."""
        with self.lock:
            self.locked_sample = self.latest if sample is None else sample
            self.live_color = None
            self.paused_at = time.monotonic()
            self.paused = True

    def resume_inference(self):
        """Discard the rejected observation before requesting a new capture."""
        with self.lock:
            if not self.paused or not self.pause_ack.is_set():
                raise RuntimeError('推理尚未确认暂停，不能重新识别')
            self.latest = self.locked_sample = None
            self.live_color = None
            self.timings = None
            self.started = time.monotonic()
            self.pause_ack.clear()
            self.paused = False

    def set_detection_interval(self, seconds):
        """Minimum seconds between detections at full speed (0 = unlimited)."""
        seconds = float(seconds)
        if not np.isfinite(seconds) or not 0. <= seconds <= 1.:
            raise ValueError('检测间隔必须在 0–1 秒之间')
        self.min_interval = seconds

    def wait_until_paused(self, timeout=10., poll=None):
        """Wait for in-flight inference/plotting to finish, keeping stop keys live."""
        deadline = time.monotonic() + timeout
        while True:
            if poll is not None:
                poll()
            self.pump()
            if self.pause_ack.is_set():
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('等待 YOLO 暂停确认超时，拒绝继续运动')
            self.pause_ack.wait(.02)

    def _capture(self):
        try:
            with self.camera_factory() as camera:
                # CUDA/model initialization can take seconds. Its image must not
                # become a supposedly fresh observation or trigger runtime age checks.
                self.stage = 'Warming up YOLO - arm has not started'
                print('D405已启动，正在预热YOLO；预热结束后重新采集新帧。', flush=True)
                warmup_start = time.monotonic()
                warmup = camera.capture(discard=0)
                self.model.predict(source=warmup.bgr, device=self.device, imgsz=self.imgsz,
                                   conf=.6, verbose=False)[0].plot()
                print(f'YOLO预热完成，用时{time.monotonic()-warmup_start:.1f}秒。', flush=True)
                self.release_cuda_cache()
                if self.inference_threads is not None:
                    # Ultralytics configures its backend on the first predict call.
                    # Cap CPU workers after warm-up so CAN retains scheduling time.
                    import torch
                    torch.set_num_threads(self.inference_threads)
                self.stage = 'Waiting for fresh detection'
                while not self.done.is_set():
                    with self.lock:
                        paused = self.paused
                        if paused:
                            self.pause_ack.set()
                    if paused:
                        if self.show_window:
                            stamp = time.monotonic()
                            bgr = camera.capture_color()
                            with self.lock:
                                if self.paused:
                                    self.live_color = (stamp, bgr)
                        # Limit display capture to at most 15 Hz to leave CPU for CAN.
                        self.done.wait(1 / 15)
                        continue
                    started = time.monotonic()
                    frame = camera.capture(discard=0)
                    captured = time.monotonic()
                    if self.paused:
                        continue
                    prediction = self.model.predict(source=frame.bgr, device=self.device, imgsz=self.imgsz,
                                                    conf=.6, verbose=False)[0]
                    annotated = prediction.plot()
                    completed = time.monotonic()
                    if self.done.is_set():
                        break
                    with self.lock:
                        self.timings = (captured-started, completed-captured, completed)
                        self.latest = (started, frame, prediction, annotated)
                    if self.min_interval > 0:
                        self.done.wait(max(0., self.min_interval-(time.monotonic()-started)))
        except BaseException as error:
            with self.lock:
                self.error = error

    def sample(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError(f'相机预览失败：{self.error}') from self.error
            return self.locked_sample if self.paused else self.latest

    def pump(self, allow_window_close=False):
        sample = self.sample()
        now = time.monotonic()
        if sample is None:
            if now-self.started > self.startup_timeout:
                raise TimeoutError(f'相机/YOLO启动超过{self.startup_timeout:g}秒，尚未获得预热后的新检测帧')
            if not self.show_window:
                return
            canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        else:
            stamp, _, _, annotated = sample
            if not self.paused and now-stamp > 3:
                with self.lock:
                    timings = self.timings
                detail = '' if timings is None else (
                    f'；上一帧采集{timings[0]:.2f}秒，推理/绘制{timings[1]:.2f}秒，'
                    f'距结果发布{now-timings[2]:.2f}秒')
                raise TimeoutError(f'相机检测帧年龄{now-stamp:.1f}秒，超过3秒{detail}')
            if not self.show_window:
                return
            if now-self.last_draw < self.display_interval:
                self._window_events(allow_window_close)
                return
            canvas = annotated.copy()
        display_status = ''
        with self.lock:
            paused, live_color, paused_at = self.paused, self.live_color, self.paused_at
        if paused and self.show_window:
            stamp = paused_at if live_color is None else live_color[0]
            if now-stamp > 3:
                raise TimeoutError('相机实时画面超过3秒未更新')
            if live_color is not None:
                canvas = live_color[1].copy()
                display_status = 'LIVE RGB | target locked | YOLO paused'
            else:
                display_status = 'Waiting for live RGB | target locked'
        for index, text in enumerate((self.stage, self.detail, display_status,
                                     'Type go in TERMINAL; Q/Esc/Space = STOP')):
            y = 24 + index*26
            self.cv2.putText(canvas, text, (10, y), self.cv2.FONT_HERSHEY_SIMPLEX,
                                 .55, (0, 0, 0), 4)
            self.cv2.putText(canvas, text, (10, y), self.cv2.FONT_HERSHEY_SIMPLEX,
                             .55, (0, 255, 255), 1)
        self.cv2.imshow(self.title, canvas)
        self.last_draw = now
        self._window_events(allow_window_close)

    def _window_events(self, allow_window_close=False):
        key = self.cv2.waitKey(1) & 0xff
        if key in (ord('q'), ord('Q'), 27, 32):
            raise KeyboardInterrupt('相机窗口停止键')
        if self.cv2.getWindowProperty(self.title, self.cv2.WND_PROP_VISIBLE) < 1:
            if allow_window_close:
                self.show_window = False
                return
            raise KeyboardInterrupt('相机窗口已关闭')

    def close(self):
        self.done.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
        if self.show_window:
            self.cv2.destroyAllWindows()
