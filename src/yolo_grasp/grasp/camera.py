from dataclasses import dataclass
import json
from pathlib import Path
import time
import numpy as np


@dataclass
class Frame:
    bgr: np.ndarray
    depth_m: np.ndarray
    points: np.ndarray  # H x W x 3, aligned colour camera coordinates, metres
    metadata: dict

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory / "frame.npz", bgr=self.bgr,
                            depth_m=self.depth_m, points=self.points)
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2))
        import cv2
        if not cv2.imwrite(str(directory / "color.png"), self.bgr):
            raise OSError("Could not save color image")

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        with np.load(directory / "frame.npz", allow_pickle=False) as a:
            frame = cls(a["bgr"], a["depth_m"], a["points"],
                        json.loads((directory / "metadata.json").read_text()))
        frame.validate()
        return frame

    def validate(self):
        h, w = self.depth_m.shape
        if self.bgr.shape != (h, w, 3) or self.points.shape != (h, w, 3):
            raise ValueError("RGB/depth/point grid dimensions differ")


class RealSense:
    def __init__(self, cfg):
        self.cfg = cfg
        self.pipeline = None

    def __enter__(self):
        import pyrealsense2 as rs
        self.rs = rs
        config = rs.config()
        config.enable_device(self.cfg["serial"])
        for key, stream, fmt in (("color", rs.stream.color, rs.format.bgr8),
                                 ("depth", rs.stream.depth, rs.format.z16)):
            s = self.cfg[key]
            config.enable_stream(stream, s["width"], s["height"], fmt, s["fps"])
        pipeline = rs.pipeline()
        profile = pipeline.start(config)
        self.pipeline = pipeline
        try:
            if self.cfg.get('global_time', False):
                for sensor in profile.get_device().query_sensors():
                    if sensor.supports(rs.option.global_time_enabled):
                        sensor.set_option(rs.option.global_time_enabled, 1)
            self.scale = profile.get_device().first_depth_sensor().get_depth_scale()
            self.align = rs.align(rs.stream.color)
            self.pointcloud = rs.pointcloud()
            self.capture(discard=self.cfg.get("warmup_frames", 30))
        except BaseException:
            pipeline.stop()
            self.pipeline = None
            raise
        return self

    def capture_color(self):
        """Read display-only colour without depth alignment or point-cloud work."""
        if self.pipeline is None:
            raise RuntimeError("Camera is not started")
        frames = self.pipeline.wait_for_frames(self.cfg.get("timeout_ms", 5000))
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("Missing color frame")
        return np.asanyarray(color.get_data()).copy()

    def capture(self, discard=5):
        if isinstance(discard, bool) or not isinstance(discard, int) or discard < 0:
            raise ValueError("Discard count must be a nonnegative integer")
        if self.pipeline is None:
            raise RuntimeError("Camera is not started")
        # Discard buffered images only after the arm has been confirmed stationary.
        for _ in range(discard + 1):
            raw = self.pipeline.wait_for_frames(self.cfg.get("timeout_ms", 5000))
        color_raw, depth_raw = raw.get_color_frame(), raw.get_depth_frame()
        if not color_raw or not depth_raw:
            raise RuntimeError("Incomplete RGB-D frameset")
        if color_raw.get_frame_timestamp_domain() != depth_raw.get_frame_timestamp_domain():
            raise RuntimeError("RGB and depth timestamps use different clock domains")
        skew = abs(color_raw.get_timestamp() - depth_raw.get_timestamp())
        if skew > self.cfg.get("max_rgb_depth_skew_ms", 35):
            raise RuntimeError(f"RGB/depth timestamp skew {skew:.1f} ms")
        frames = self.align.process(raw)
        depth, color = frames.get_depth_frame(), frames.get_color_frame()
        if not depth or not color:
            raise RuntimeError("Alignment returned an incomplete RGB-D frameset")
        intr = depth.profile.as_video_stream_profile().intrinsics
        # librealsense handles the actual distortion model during deprojection.
        cloud = self.pointcloud.calculate(depth)
        points = np.asanyarray(cloud.get_vertices()).view(np.float32).reshape(intr.height, intr.width, 3).copy()
        color_intr = color_raw.profile.as_video_stream_profile().intrinsics
        metadata = {"color_intrinsics": {"width": color_intr.width, "height": color_intr.height,
                    "fx": color_intr.fx, "fy": color_intr.fy, "ppx": color_intr.ppx,
                    "ppy": color_intr.ppy, "model": str(color_intr.model), "coeffs": list(color_intr.coeffs)},
                    "serial": self.cfg["serial"], "host_monotonic_s": time.monotonic(),
                    "depth_scale_m": self.scale, "color_timestamp_ms": color_raw.get_timestamp(),
                    "depth_timestamp_ms": depth_raw.get_timestamp(),
                    "timestamp_domain": str(depth_raw.get_frame_timestamp_domain()),
                    "frame_number": depth_raw.get_frame_number(),
                    "streams": {name: {"width": f.profile.as_video_stream_profile().width(),
                                       "height": f.profile.as_video_stream_profile().height(),
                                       "fps": f.profile.fps(), "format": str(f.profile.format())}
                                for name, f in (("color", color_raw), ("depth", depth_raw))},
                    "intrinsics": {"width": intr.width, "height": intr.height,
                                   "fx": intr.fx, "fy": intr.fy, "ppx": intr.ppx,
                                   "ppy": intr.ppy, "model": str(intr.model), "coeffs": list(intr.coeffs)}}
        frame = Frame(np.asanyarray(color.get_data()).copy(),
                      np.asanyarray(depth.get_data()).astype(float) * self.scale, points, metadata)
        frame.validate()
        return frame

    def __exit__(self, *args):
        if self.pipeline is not None:
            self.pipeline.stop()
            self.pipeline = None
