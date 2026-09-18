from types import SimpleNamespace
import numpy as np
import pytest
from grasp.hover_preview import HoverPreview


class CV:
    FONT_HERSHEY_SIMPLEX = 0
    WND_PROP_VISIBLE = 0
    key = -1
    visible = 1
    shown = None
    def putText(self, *args): pass
    def imshow(self, title, canvas): self.shown = canvas
    def waitKey(self, delay): return self.key
    def getWindowProperty(self, *args): return self.visible


@pytest.mark.parametrize('key,visible', [(ord('q'), 1), (27, 1), (32, 1), (-1, 0)])
def test_window_stop(key, visible):
    cv = CV(); cv.key = key; cv.visible = visible
    preview = HoverPreview(None, None, cv)
    with pytest.raises(KeyboardInterrupt):
        preview.pump()


def test_worker_failure_propagates_to_control():
    preview = HoverPreview(None, None, CV())
    preview.error = RuntimeError('camera disconnected')
    with pytest.raises(RuntimeError, match='camera disconnected'):
        preview.pump()


def test_detection_interval_can_be_capped_while_moving():
    """The placement leg detects during motion; the cap protects the CAN thread."""
    preview = HoverPreview(None, None, CV())
    assert preview.min_interval == 0.
    preview.set_detection_interval(.25)
    assert preview.min_interval == .25
    preview.set_detection_interval(0.)
    assert preview.min_interval == 0.
    for bad in (-.1, 1.5, float('nan')):
        with pytest.raises(ValueError):
            preview.set_detection_interval(bad)


def test_stale_preview_rejected(monkeypatch):
    preview = HoverPreview(None, None, CV())
    preview.latest = (10, None, None, np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 14)
    with pytest.raises(TimeoutError):
        preview.pump()


def test_display_throttling_still_polls_stop_keys(monkeypatch):
    cv = CV(); preview = HoverPreview(None, None, cv)
    preview.display_interval = .1
    preview.latest = (10, None, None, np.zeros((4, 4, 3), dtype=np.uint8))
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 10)
    preview.pump()
    drawn = cv.shown
    cv.key = 27
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 10.01)
    with pytest.raises(KeyboardInterrupt):
        preview.pump()
    assert cv.shown is drawn


def test_preview_does_not_mutate_observation(monkeypatch):
    cv = CV(); preview = HoverPreview(None, None, cv)
    original = np.zeros((4, 4, 3), dtype=np.uint8)
    preview.latest = (10, None, None, original)
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 10)
    preview.pump()
    assert cv.shown is not original
    assert np.array_equal(cv.shown, original)


def test_slow_warmup_is_discarded_and_new_frame_published(monkeypatch):
    from types import SimpleNamespace
    clock = [0.]
    frames = []
    preview = HoverPreview(None, None, CV())
    class Camera:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def capture(self, discard):
            frame = SimpleNamespace(bgr=np.zeros((4, 4, 3), dtype=np.uint8))
            frames.append(frame)
            return frame
    class Model:
        def predict(self, **kwargs):
            if len(frames) == 1:
                clock[0] += 5  # Cold initialization exceeds runtime limit.
                assert preview.latest is None
            else:
                clock[0] += .1
            return [SimpleNamespace(plot=lambda: kwargs['source'])]
    preview.camera_factory = Camera
    preview.model = Model()
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: clock[0])
    checks = iter([False, False, True])
    monkeypatch.setattr(preview.done, 'is_set', lambda: next(checks))
    preview._capture()
    assert preview.error is None
    stamp, frame, _, _ = preview.sample()
    assert len(frames) == 2 and frame is frames[1] and stamp == 5
    preview.pump()  # Fresh post-warmup image is accepted.


def test_startup_wait_displays_window_and_has_deadline(monkeypatch):
    cv = CV(); preview = HoverPreview(None, None, cv)
    preview.started = 0
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 5)
    preview.pump()
    assert cv.shown.shape == (480, 640, 3)
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 31)
    with pytest.raises(TimeoutError, match='启动超过30秒'):
        preview.pump()


def test_paused_preview_stops_inference_but_displays_live_color(monkeypatch):
    """After the hover target is locked, YOLO must stop competing with the CAN reader."""
    calls = []
    preview = HoverPreview(None, None, CV())

    class Camera:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def capture(self, discard):
            return SimpleNamespace(bgr=np.zeros((4, 4, 3), dtype=np.uint8))
        def capture_color(self):
            return np.full((4, 4, 3), 42, dtype=np.uint8)

    class Model:
        def predict(self, **kwargs):
            calls.append(kwargs.get('imgsz'))
            return [SimpleNamespace(plot=lambda: kwargs['source'])]

    preview.camera_factory = Camera
    preview.model = Model()
    locked = (0, None, None, np.ones((4, 4, 3), dtype=np.uint8))
    preview.pause_inference(locked)
    checks = iter([False, True])          # one paused iteration, then stop
    monkeypatch.setattr(preview.done, 'is_set', lambda: next(checks))
    preview._capture()
    assert len(calls) == 1, 'only the warmup may run inference once paused'
    assert preview.pause_ack.is_set()
    assert preview.sample() is locked
    preview.pump()
    assert np.all(preview.cv2.shown == 42)
    assert np.all(locked[3] == 1), 'display frames must never replace the planned observation'


def test_paused_live_preview_staleness_is_rejected(monkeypatch):
    preview = HoverPreview(None, None, CV())
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 10)
    preview.pause_inference((0, None, None, np.zeros((4, 4, 3))))
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 14)
    with pytest.raises(TimeoutError, match='实时画面'):
        preview.pump()
    preview.live_color = (14, np.ones((4, 4, 3)))
    preview.pump()
    monkeypatch.setattr('grasp.hover_preview.time.monotonic', lambda: 18)
    with pytest.raises(TimeoutError, match='实时画面'):
        preview.pump()


def test_color_capture_uses_only_color_and_copies_buffer():
    from grasp.camera import RealSense
    camera = RealSense({})
    source = np.ones((4, 4, 3), dtype=np.uint8)
    color = SimpleNamespace(get_data=lambda: source)
    camera.pipeline = SimpleNamespace(wait_for_frames=lambda timeout:
                                     SimpleNamespace(get_color_frame=lambda: color))
    result = camera.capture_color()
    assert np.array_equal(result, source)
    assert not np.shares_memory(result, source)


def test_pause_ack_waits_for_inflight_prediction_and_preserves_target():
    import threading
    entered = threading.Event()
    release = threading.Event()
    preview = HoverPreview(None, None, CV(), device='cpu', show_window=False)
    locked = (0, None, None, np.ones((4, 4, 3), dtype=np.uint8))
    calls = []

    class Camera:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def capture(self, discard):
            return SimpleNamespace(bgr=np.zeros((4, 4, 3), dtype=np.uint8))

    class Model:
        def predict(self, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                entered.set()
                assert release.wait(2)
            return [SimpleNamespace(plot=lambda: kwargs['source'])]

    preview.camera_factory = Camera
    preview.model = Model()
    preview.start()
    try:
        assert entered.wait(2)
        preview.pause_inference(locked)
        assert not preview.pause_ack.is_set()
        assert preview.sample() is locked
        release.set()
        polls = []
        preview.wait_until_paused(timeout=2, poll=lambda: polls.append(1))
        assert polls and preview.pause_ack.is_set()
        assert len(calls) == 2
        assert preview.sample() is locked
        preview.pump()  # Frozen image does not masquerade as a fresh detection.
    finally:
        release.set()
        preview.close()


def test_pause_wait_times_out_and_checks_stop_keys():
    preview = HoverPreview(None, None, CV(), show_window=False)
    preview.pause_inference((0, None, None, np.zeros((4, 4, 3))))
    with pytest.raises(TimeoutError, match='暂停确认超时'):
        preview.wait_until_paused(timeout=0)
    def stop():
        raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        preview.wait_until_paused(poll=stop)


def test_window_close_after_arrival_disables_preview_without_stop():
    cv = CV(); cv.visible = 0
    preview = HoverPreview(None, None, cv)
    preview.pause_inference((0, None, None, np.zeros((4, 4, 3), dtype=np.uint8)))
    preview.pump(allow_window_close=True)
    assert not preview.show_window


def test_explicit_window_stop_after_arrival_is_preserved():
    cv = CV(); cv.key = ord('q')
    preview = HoverPreview(None, None, cv)
    with pytest.raises(KeyboardInterrupt):
        preview.pump(allow_window_close=True)


def test_resume_discards_rejected_and_inflight_samples():
    preview = HoverPreview(None, None, CV())
    preview.latest = (1, None, None, None)
    preview.pause_inference()
    with pytest.raises(RuntimeError, match='尚未确认暂停'):
        preview.resume_inference()
    preview.pause_ack.set()
    preview.latest = (2, None, None, None)
    preview.resume_inference()
    assert preview.sample() is None
    assert preview.locked_sample is None
    assert not preview.pause_ack.is_set() and not preview.paused
    # A second pause must await a new acknowledgement.
    preview.pause_inference()
    with pytest.raises(RuntimeError, match='尚未确认暂停'):
        preview.resume_inference()
