from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
from grasp.piper_model import piper_planning_limits


def test_planning_limits_use_ordinary_piper_with_official_j6_bound():
    limits = np.asarray(piper_planning_limits(ROBOT_JOINT_LIMIT_PRESET_RAD))
    ordinary = np.asarray(list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper'].values()))

    np.testing.assert_allclose(limits[:5], ordinary[:5])
    np.testing.assert_allclose(np.rad2deg(limits[5]), [-120.0, 120.0])
    np.testing.assert_allclose(np.rad2deg(limits[3:5, 1]), [100.0, 70.0], atol=1e-4)


def test_planning_limits_returns_fresh_mutable_values():
    first = piper_planning_limits(ROBOT_JOINT_LIMIT_PRESET_RAD)
    second = piper_planning_limits(ROBOT_JOINT_LIMIT_PRESET_RAD)
    first[0][0] = 0.0
    assert second[0][0] != 0.0
