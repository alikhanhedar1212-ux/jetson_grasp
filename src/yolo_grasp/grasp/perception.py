from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from .geometry import apply, table_basis


class DetectionError(RuntimeError):
    pass


class YoloDetector:
    def __init__(self, model_path, cfg):
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"Local red-block model missing: {model_path}")
        from ultralytics import YOLO
        self.model = YOLO(str(model_path), task="detect")
        self.cfg = cfg

    def detect(self, bgr):
        result = self.model.predict(bgr, conf=self.cfg["confidence"],
                                    device=self.cfg["device"], verbose=False)[0]
        boxes = result.boxes
        names = result.names
        if self.cfg["class_name"] not in names.values():
            raise DetectionError("Model does not contain the configured red-block class")
        found = [b.xyxy[0].cpu().numpy() for b in boxes
                 if names[int(b.cls.item())] == self.cfg["class_name"]]
        if len(found) != 1:
            raise DetectionError(f"Expected one target, got {len(found)}")
        return found[0]


@dataclass
class Target:
    center: np.ndarray
    top: np.ndarray
    height: float
    axis: np.ndarray
    normal: np.ndarray
    footprint: np.ndarray
    yaw: float


def estimate(frame, box, T_base_camera, table, cfg):
    frame.validate()
    h, w = frame.depth_m.shape
    box = np.asarray(box, dtype=float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise DetectionError("Invalid detection box")
    x1, y1 = np.maximum(np.floor(box[:2]).astype(int), 0)
    x2, y2 = np.minimum(np.ceil(box[2:]).astype(int), [w, h])
    if x2 <= x1 or y2 <= y1:
        raise DetectionError("Empty detection box")
    hsv = cv2.cvtColor(frame.bgr, cv2.COLOR_BGR2HSV)
    mask = (((hsv[..., 0] <= cfg["red_hue_low_max"]) |
             (hsv[..., 0] >= cfg["red_hue_high_min"])) &
            (hsv[..., 1] >= cfg["red_saturation_min"]) &
            (hsv[..., 2] >= cfg["red_value_min"]))
    roi = np.zeros((h, w), dtype=bool)
    roi[y1:y2, x1:x2] = True
    full_red_mask = mask.copy()
    mask &= roi
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    candidates = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= cfg["min_pixels"]]
    if len(candidates) != 1:
        raise DetectionError("Red mask must contain one sufficiently large connected component")
    component = labels == candidates[0]
    border = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
    if np.any(border & ~roi & full_red_mask):
        raise DetectionError("Detection box clips a continuing red component")
    if component[0].any() or component[-1].any() or component[:, 0].any() or component[:, -1].any():
        raise DetectionError("Target is clipped by the image boundary")
    valid = (component & np.isfinite(frame.points).all(axis=2)
             & np.isfinite(frame.depth_m) & (frame.depth_m >= cfg["min_depth_m"])
             & (frame.depth_m <= cfg["max_depth_m"]) & (frame.points[..., 2] > 0))
    if valid.sum() < cfg["min_pixels"] or valid.sum() / component.sum() < cfg["min_valid_fraction"]:
        raise DetectionError("Insufficient valid target depth")
    points = apply(T_base_camera, frame.points[valid])
    basis = table_basis(table["normal"])
    n = basis[:, 2]
    heights = points @ n + table["offset_m"]
    # Test each allowed top height. Side faces and table pixels are excluded.
    hypotheses = []
    for height in (.020, .030):
        keep = abs(heights - height) <= cfg["height_tolerance_m"]
        if keep.sum() < cfg["min_pixels"]:
            continue
        top_points = points[keep]
        measured_height = float(np.median(heights[keep]))
        top_points = top_points[abs(heights[keep] - measured_height) <= cfg["top_band_m"]]
        if len(top_points) < cfg["min_pixels"]:
            continue
        xy = top_points @ basis[:, :2]
        # Remove isolated depth speckles before fitting the enclosing rectangle.
        # The radius is metric, so this behaves consistently across both cameras.
        from scipy.spatial import cKDTree
        distance, _ = cKDTree(xy).query(xy, k=4)
        xy = xy[distance[:, -1] <= cfg.get("neighbor_radius_m", .003)]
        if len(xy) < cfg["min_pixels"]:
            continue
        rectangle = cv2.minAreaRect(xy.astype(np.float32))
        (cx, cy), _, _ = rectangle
        corners = cv2.boxPoints(rectangle)
        edges = np.roll(corners, -1, axis=0) - corners
        lengths = np.linalg.norm(edges, axis=1)
        expected = np.array([.030, .030] if height == .020 else [.020, .030])
        observed = np.sort(lengths[:2])
        if np.max(abs(observed - expected)) > cfg["dimension_tolerance_m"]:
            continue
        # Reject sparse/partial outlines even when their outer rectangle fits.
        coverage = cv2.contourArea(cv2.convexHull(xy.astype(np.float32))) / max(np.prod(observed), 1e-12)
        if coverage < cfg["min_hull_coverage"]:
            continue
        edge = edges[np.argmax(lengths)]
        yaw = float(np.arctan2(edge[1], edge[0]) % (np.pi / 2 if height == .020 else np.pi))
        axis = basis[:, 0] * np.cos(yaw) + basis[:, 1] * np.sin(yaw)
        top = basis[:, :2] @ [cx, cy] + n * (measured_height - table["offset_m"])
        hypotheses.append(Target(top - n * height / 2, top, height, axis, n,
                                 expected, yaw))
    if len(hypotheses) != 1:
        raise DetectionError("Top height/footprint is missing or ambiguous; reacquire target")
    return hypotheses[0]
