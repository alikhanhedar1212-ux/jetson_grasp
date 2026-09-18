from dataclasses import dataclass
import cv2
import numpy as np
from .geometry import table_basis, transform


@dataclass
class Plan:
    grasp: np.ndarray
    pregrasp: np.ndarray
    place: np.ndarray
    preplace: np.ndarray
    open_width: float
    expected_width: float
    region_index: int


class WorkspaceError(ValueError):
    """A well-formed pose lies outside the configured endpoint bounds."""


def check_workspace(t, cfg):
    t = transform(t)
    low, high = np.asarray(cfg["workspace_min_m"]), np.asarray(cfg["workspace_max_m"])
    if low.shape != (3,) or high.shape != (3,) or not np.all(low < high):
        raise ValueError("Invalid measured workspace bounds")
    if not np.all((t[:3, 3] >= low) & (t[:3, 3] <= high)):
        raise WorkspaceError("TCP/flange target outside configured workspace")


def inside_region(center_uv, axis_uv, footprint, region, margin):
    polygon = np.asarray(region["polygon_uv_m"], dtype=np.float32)
    if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
        raise ValueError("Placement polygon must contain finite table-plane coordinates")
    if not cv2.isContourConvex(polygon):
        raise ValueError("Placement polygon must be ordered and convex")
    long_axis = np.asarray(axis_uv, dtype=float)
    short_axis = np.array([-long_axis[1], long_axis[0]])
    for a in (-1, 1):
        for b in (-1, 1):
            corner = np.asarray(center_uv) + a * long_axis * footprint[1] / 2 + b * short_axis * footprint[0] / 2
            if cv2.pointPolygonTest(polygon, tuple(map(float, corner)), True) < margin:
                return False
    return True


def make_plan(target, calib, cfg):
    basis = table_basis(calib["table"]["normal"])
    normal = target.normal
    grasp = np.eye(4)
    # Calibrated TCP: X closes the jaws; Z points down the approach direction.
    grasp[:3, :3] = np.column_stack([np.cross(normal, target.axis), target.axis, -normal])
    grasp[:3, 3] = target.center
    pregrasp = grasp.copy()
    pregrasp[:3, 3] += normal * cfg["approach_clearance_m"]
    width = float(target.footprint[0])
    opening = width + cfg["opening_margin_m"]
    if opening > cfg["max_gripper_width_m"]:
        raise ValueError("Required opening exceeds measured gripper width")
    actual_center_height = float(np.dot(target.center, normal) + calib["table"]["offset_m"])
    if cfg["finger_below_tcp_m"] + cfg["table_clearance_m"] >= actual_center_height:
        raise ValueError("Finger geometry does not clear the table at object-centre grasp")
    inverse_tcp = np.linalg.inv(calib["T_flange_tcp"])
    for t in (grasp, pregrasp):
        check_workspace(t, cfg)
        check_workspace(t @ inverse_tcp, cfg)
    for index, region in enumerate(calib["place_regions"]):
        uv = np.asarray(region["center_uv_m"], dtype=float)
        if not inside_region(uv, basis[:, :2].T @ target.axis, target.footprint,
                             region, cfg["placement_margin_m"]):
            continue
        place = grasp.copy()
        place[:3, 3] = basis[:, :2] @ uv + normal * (
            target.height / 2 - calib["table"]["offset_m"] + cfg["release_clearance_m"])
        preplace = place.copy()
        preplace[:3, 3] += normal * cfg["approach_clearance_m"]
        try:
            for t in (place, preplace):
                check_workspace(t, cfg)
                # Endpoints only; this does not check links or swept volume.
                check_workspace(t @ inverse_tcp, cfg)
        except WorkspaceError:
            continue
        return Plan(grasp, pregrasp, place, preplace, opening, width, index)
    raise ValueError("No placement region fits the object and endpoint workspace limits")


def accept_refinement(coarse, fine, cfg):
    if coarse.height != fine.height:
        raise ValueError("Wrist camera disagrees with global target height")
    if np.linalg.norm(coarse.center - fine.center) > cfg["max_correction_m"]:
        raise ValueError("Wrist target does not match global target position")
    period = np.pi / 2 if fine.height == .020 else np.pi
    delta = (fine.yaw - coarse.yaw + period / 2) % period - period / 2
    if abs(delta) > cfg["max_correction_rad"]:
        raise ValueError("Wrist correction exceeds angle limit")
    # Choose the equivalent rectangle orientation nearest the coarse orientation.
    n = coarse.normal
    fine.axis = np.cos(delta) * coarse.axis + np.sin(delta) * np.cross(n, coarse.axis)
    fine.yaw = coarse.yaw + delta
    return fine
