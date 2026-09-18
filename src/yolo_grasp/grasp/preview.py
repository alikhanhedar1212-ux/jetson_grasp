"""Standalone RGB dataset acquisition; no model, calibration or robot connection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def add_arguments(parser):
    parser.add_argument("--serial", default="243222074879")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "rgb",
                        help="Output root (default: source project data/rgb); each launch creates a separate batch")
    parser.add_argument("--timeout-ms", type=positive_int, default=5000)
    parser.add_argument("--note", default="", help="Scene/lighting notes for this batch")


class Batch:
    def __init__(self, root, serial, note):
        self.path = Path(root) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "images").mkdir()
        (self.path / "metadata").mkdir()
        self.count = 0
        self.serial = serial
        (self.path / "batch.json").write_text(json.dumps({
            "schema_version": 1, "batch_id": self.path.name, "started_at_utc": utc_now(),
            "serial": serial, "note": note,
            "stream": {"width": 640, "height": 480, "fps": 30, "format": "bgr8"},
            "content": "raw RGB only; no depth or annotations",
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(self, bgr, metadata):
        import cv2
        stem = f"{self.count + 1:06d}"
        image_path = self.path / "images" / f"{stem}.png"
        metadata_path = self.path / "metadata" / f"{stem}.json"
        try:
            if not cv2.imwrite(str(image_path), bgr):
                raise OSError(f"Could not save {image_path}")
            metadata_path.write_text(json.dumps({
                **metadata, "serial": self.serial, "saved_at_utc": utc_now(),
                "batch_id": self.path.name, "image": f"images/{stem}.png",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except BaseException:
            image_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise
        self.count += 1
        return image_path


def run(args):
    import cv2
    import numpy as np
    import pyrealsense2 as rs

    if not args.serial.strip():
        raise ValueError("Camera serial must not be empty")
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline = rs.pipeline()
    started = False
    window = "D435i RGB | S: save | Q/Esc: quit"
    try:
        pipeline.start(config)
        started = True
        # Let exposure settle before presenting training images.
        for _ in range(30):
            pipeline.wait_for_frames(args.timeout_ms)
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        batch = Batch(args.output, args.serial, args.note)
        print(f"采集目录: {batch.path.resolve()}\n点击预览窗口，S 保存，Q/Esc 退出。", flush=True)
        last = time.monotonic()
        fps = 0.
        while True:
            color = pipeline.wait_for_frames(args.timeout_ms).get_color_frame()
            if not color:
                raise RuntimeError("Camera returned no color frame")
            raw = np.asanyarray(color.get_data()).copy()
            captured_at = utc_now()
            now = time.monotonic()
            instantaneous = 1. / max(now - last, 1e-9)
            fps = instantaneous if fps == 0 else .9 * fps + .1 * instantaneous
            last = now
            display = raw.copy()
            cv2.putText(display, f"FPS {fps:.1f}  Saved {batch.count}  S: save  Q: quit",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 0), 1)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S")):
                path = batch.save(raw, {
                    "captured_at_utc": captured_at, "host_monotonic_s": now,
                    "frame_number": color.get_frame_number(),
                    "camera_timestamp_ms": color.get_timestamp(),
                    "timestamp_domain": str(color.get_frame_timestamp_domain()),
                    "width": raw.shape[1], "height": raw.shape[0],
                })
                print(f"已保存 {batch.count}: {path}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 0
    finally:
        try:
            if started:
                pipeline.stop()
        finally:
            cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
