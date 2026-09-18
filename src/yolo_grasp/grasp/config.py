import json
from pathlib import Path
import numpy as np
from .geometry import transform, table_basis


def load(path, require_cameras=True):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported config schema")
    cfg["_root"] = path.parent
    if not require_cameras:
        return cfg
    serials = [cfg["cameras"][role]["serial"] for role in ("global", "wrist")]
    if any(not isinstance(s, str) or not s.strip() or s.startswith("REPLACE_") for s in serials) or len(set(serials)) != 2:
        raise ValueError("Configure two distinct camera serial numbers")
    validate_cameras(cfg)
    return cfg


def number(section, key, minimum=0, maximum=None, integer=False, inclusive=False):
    value = section.get(key)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not np.isfinite(value) or (integer and not isinstance(value, int))
            or (value < minimum if inclusive else value <= minimum)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"Missing or invalid configuration: {key}")
    return value


def validate_cameras(cfg):
    for camera in cfg["cameras"].values():
        for stream in ("color", "depth"):
            for key in ("width", "height", "fps"):
                number(camera[stream], key, integer=True)
        number(camera, "warmup_frames", integer=True, inclusive=True)
        number(camera, "timeout_ms", integer=True)
        number(camera, "max_rgb_depth_skew_ms", inclusive=True)


def validate_perception(cfg):
    p = cfg["perception"]
    number(p, "min_pixels", minimum=4, inclusive=True, integer=True)
    for key in ("min_valid_fraction", "min_hull_coverage"):
        number(p, key, maximum=1)
    for key in ("height_tolerance_m", "top_band_m", "dimension_tolerance_m"):
        number(p, key)
    if p["height_tolerance_m"] >= .005:
        raise ValueError("Height tolerance must distinguish 20 and 30 mm hypotheses")
    number(p, "min_depth_m")
    number(p, "max_depth_m", minimum=p["min_depth_m"])
    for key in ("red_hue_low_max", "red_hue_high_min"):
        number(p, key, maximum=179, inclusive=True, integer=True)
    if p["red_hue_low_max"] >= p["red_hue_high_min"]:
        raise ValueError("Red hue ranges overlap")
    for key in ("red_saturation_min", "red_value_min"):
        number(p, key, maximum=255, inclusive=True, integer=True)
    if "neighbor_radius_m" in p:
        number(p, "neighbor_radius_m")
    number(cfg["detector"], "confidence", maximum=1)
    if not isinstance(cfg["detector"].get("class_name"), str) or not cfg["detector"]["class_name"].strip():
        raise ValueError("Configure detector.class_name")


def validate_feedback(cfg):
    # Iterate required fields, not supplied keys: omitted fields must fail BEFORE TX.
    for key in ("max_feedback_age_s", "max_pose_skew_s", "position_tolerance_m",
                "angle_tolerance_rad", "settle_s", "motion_timeout_s"):
        number(cfg["feedback"], key)
    if cfg["feedback"]["motion_timeout_s"] <= cfg["feedback"]["settle_s"]:
        raise ValueError("Motion timeout must exceed settling duration")


def local_path(cfg, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else cfg["_root"] / path


def calibration(cfg):
    data = json.loads(local_path(cfg, cfg["calibration_file"]).read_text())
    if data.get("validated") is not True:
        raise ValueError("Calibration must pass independent position validation first")
    for key in ("T_base_global", "T_flange_wrist", "T_flange_tcp"):
        data[key] = transform(data[key])
    for role in ("global", "wrist"):
        if data["camera_serials"][role] != cfg["cameras"][role]["serial"]:
            raise ValueError(f"Calibration camera identity mismatch: {role}")
    table_basis(data["table"]["normal"])
    if not np.isfinite(data["table"]["offset_m"]):
        raise ValueError("Invalid table plane offset")
    if not data.get("place_regions"):
        raise ValueError("At least one measured placement region is required")
    from .planning import inside_region
    for region in data["place_regions"]:
        uv = np.asarray(region["center_uv_m"], dtype=float)
        if uv.shape != (2,) or not np.isfinite(uv).all():
            raise ValueError("Invalid placement center")
        if not inside_region(uv, [1., 0.], [0., 0.], region, 0):
            raise ValueError("Placement center lies outside its region")
    return data


def validate_motion(cfg, require_execution=True):
    validate_perception(cfg)
    g = cfg["grasp"]
    for name in ("approach_clearance_m", "opening_margin_m", "max_gripper_width_m",
                 "table_clearance_m", "placement_margin_m", "release_clearance_m",
                 "max_correction_m", "max_correction_rad"):
        number(g, name)
    number(g, "finger_below_tcp_m", inclusive=True)
    low, high = np.asarray(g["workspace_min_m"], dtype=float), np.asarray(g["workspace_max_m"], dtype=float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or not np.all(low < high):
        raise ValueError("Configure measured workspace bounds")
    if not require_execution:
        return
    rc = cfg["robot"]
    if rc["model"] != "piper_x" or not rc.get("firmware"):
        raise ValueError("Piper X model and actual firmware are required")
    if rc.get("motion_enabled") is not True or rc.get("paths_verified") is not True:
        raise ValueError("Motion is disabled or physical transfer paths are unverified")
    number(rc, "speed_percent", maximum=100, integer=True)
    for key in ("gripper_force_n", "min_hold_force_n", "gripper_tolerance_m"):
        number(rc, key)
    if rc["min_hold_force_n"] > rc["gripper_force_n"] or rc["gripper_force_n"] > 32.767:
        raise ValueError("Gripper force thresholds exceed commanded force or signed feedback range")
    if rc["gripper_tolerance_m"] >= .01:
        raise ValueError("Gripper tolerance must distinguish a 20 mm object from empty closure")
    validate_feedback(cfg)
