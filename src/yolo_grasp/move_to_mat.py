"""Entry point for the taught mat pose and gripper opening sequence."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp.mat_pose import main

if __name__ == '__main__':
    raise SystemExit(main())
