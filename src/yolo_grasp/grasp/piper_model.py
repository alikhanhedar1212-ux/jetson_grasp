"""Application-level model choices for the physical ordinary Piper arm."""
import numpy as np


# pyAgxArm currently exposes +/-180 deg for ordinary-Piper J6, while AgileX's
# official Piper URDF (agx_arm_urdf commit f6642ce) declares +/-120 deg. Keep
# the conservative official bound until the physical manual/firmware limit is
# independently verified. Task envelopes such as the 90 deg hover bound are
# separate and remain in force.
PIPER_OFFICIAL_J6_LIMIT_RAD = np.deg2rad(120.0)


def piper_planning_limits(constants):
    """Return ordinary-Piper limits with the official conservative J6 bound."""
    limits = np.asarray(list(constants['piper'].values()), dtype=float).copy()
    limits[5] = (-PIPER_OFFICIAL_J6_LIMIT_RAD, PIPER_OFFICIAL_J6_LIMIT_RAD)
    return limits.tolist()
