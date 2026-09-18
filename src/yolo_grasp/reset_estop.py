"""Run with the existing yolo_grasp Conda environment."""
import sys
from pathlib import Path

# Resolve the local SDK even when this script is launched from another directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp.reset_estop import main

if __name__ == "__main__":
    raise SystemExit(main())
