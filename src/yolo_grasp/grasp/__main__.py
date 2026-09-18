import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
from datetime import datetime, timezone
import numpy as np
from .camera import Frame, RealSense
from .config import load, calibration, local_path, validate_motion, validate_perception, validate_feedback
from .geometry import pose_matrix
from .perception import YoloDetector, estimate
from .planning import make_plan


def serializable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Piper X dual RGB-D grasping (default: validate only)")
    parser.add_argument("--config", default="config/example.json")
    subs = parser.add_subparsers(dest="command")
    subs.add_parser("validate", help="Offline config/calibration validation; no hardware")
    subs.add_parser("capture", help="Capture both cameras; never connect CAN")
    from .preview import add_arguments as preview_arguments
    preview_arguments(subs.add_parser("preview", help="D435i RGB preview and S-key dataset capture; no calibration or CAN"))
    replay = subs.add_parser("replay", help="Offline detection and planning; never connect CAN")
    replay.add_argument("frame")
    replay.add_argument("--role", choices=["global", "wrist"], default="global")
    replay.add_argument("--flange-pose", help="JSON file: synchronized base flange pose [m, rad] for wrist")
    replay.add_argument("--box", nargs=4, type=float, help="Offline diagnostic bounding box, bypass YOLO")
    subs.add_parser("status", help="Receive CAN state and query firmware; no enable/move commands")
    subs.add_parser("run", help="Execute one physical pick/place cycle; requires completed local motion config")
    from .arm_console import add_arguments
    add_arguments(subs.add_parser("arm", help="Interactive enable/home/disable console; no cameras or calibration"))
    args = parser.parse_args(argv)
    command = args.command or "validate"
    if command == "preview":
        from .preview import run
        return run(args)
    if command == "arm":
        from .arm_console import main as arm_main
        return arm_main(["--channel", args.channel, "--home-file", str(args.home_file),
                         "--speed", str(args.speed)])
    cfg = load(args.config, require_cameras=command != "status")
    if command == "status":
        from .robot import inspect_piper
        print(json.dumps(inspect_piper(cfg), default=serializable, indent=2))
        return 0
    if command == "capture":
        output = new_output(cfg)
        with ExitStack() as stack:
            cameras = {role: stack.enter_context(RealSense(c)) for role, c in cfg["cameras"].items()}
            for role, camera in cameras.items():
                camera.capture().save(output / role)
        print(output)
        return 0
    calib = calibration(cfg)
    if command == "validate":
        validate_perception(cfg)
        validate_feedback(cfg)
        print("Camera identities and calibration structure validated; no hardware connected.")
        print("Motion readiness and measured accuracy are not established by this command.")
        return 0
    if command == "replay":
        validate_perception(cfg)
        frame = Frame.load(args.frame)
        if frame.metadata["serial"] != cfg["cameras"][args.role]["serial"]:
            raise ValueError("Recorded camera serial does not match config")
        t = calib["T_base_global"]
        if args.role == "wrist":
            fp = (json.loads(Path(args.flange_pose).read_text()) if args.flange_pose
                  else frame.metadata.get("flange_pose_base"))
            if fp is None:
                raise ValueError("Wrist replay requires capture-time flange pose")
            t = pose_matrix(fp) @ calib["T_flange_wrist"]
        box = args.box
        if box is None:
            detector = YoloDetector(local_path(cfg, cfg["detector"]["model"]), cfg["detector"])
            box = detector.detect(frame.bgr)
        target = estimate(frame, box, t, calib["table"], cfg["perception"])
        validate_motion(cfg, require_execution=False)
        plan = make_plan(target, calib, cfg["grasp"])
        print(json.dumps({"target": asdict(target), "plan": asdict(plan), "offline": True},
                         default=serializable, indent=2))
        return 0
    from .robot import Piper
    validate_motion(cfg)
    detector = YoloDetector(local_path(cfg, cfg["detector"]["model"]), cfg["detector"])
    output = new_output(cfg)
    (output / "config.json").write_text(json.dumps(cfg, default=serializable, indent=2))
    (output / "calibration.json").write_text(json.dumps(calib, default=serializable, indent=2))
    from .vision import Vision
    from .workflow import Workflow
    # Signal handlers only raise; normal exception unwinding performs stop/cleanup.
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")
    old = signal.signal(signal.SIGTERM, interrupted)
    try:
        with ExitStack() as stack:
            log = stack.enter_context((output / "events.jsonl").open("w", buffering=1))
            def event(value):
                line = json.dumps(value, default=serializable)
                log.write(line + "\n")
                print(line, flush=True)
            cameras = {role: stack.enter_context(RealSense(c)) for role, c in cfg["cameras"].items()}
            robot = stack.enter_context(Piper(cfg, calib))
            vision = Vision(cameras, detector, robot, calib, cfg, output)
            Workflow(robot, vision, calib, cfg["grasp"], event).run()
    finally:
        signal.signal(signal.SIGTERM, old)
    return 0


def new_output(cfg):
    path = local_path(cfg, cfg["output_dir"]) / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    path.mkdir(parents=True, exist_ok=False)
    return path


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
