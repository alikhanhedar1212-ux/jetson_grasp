"""All distances are metres; T_A_B maps coordinates from B into A."""
import numpy as np
from scipy.spatial.transform import Rotation


def transform(value):
    t = np.asarray(value, dtype=float)
    if (t.shape != (4, 4) or not np.isfinite(t).all()
            or not np.allclose(t[3], [0, 0, 0, 1], atol=1e-7)
            or not np.allclose(t[:3, :3].T @ t[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(t[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Expected a finite rigid 4x4 transform")
    return t


def pose_matrix(pose):
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (6,) or not np.isfinite(pose).all():
        raise ValueError("Pose must contain six finite values in m/rad")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", pose[3:]).as_matrix()
    t[:3, 3] = pose[:3]
    return t


def matrix_pose(t):
    t = transform(t)
    return np.r_[t[:3, 3], Rotation.from_matrix(t[:3, :3]).as_euler("xyz")].tolist()


def apply(t, points):
    t = transform(t)
    return np.asarray(points) @ t[:3, :3].T + t[:3, 3]


def pose_error(actual, target):
    a, b = transform(actual), transform(target)
    return (float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
            float(Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).magnitude()))


def table_basis(normal):
    n = np.asarray(normal, dtype=float)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.isclose(np.linalg.norm(n), 1):
        raise ValueError("Table normal must be a unit vector pointing above the table")
    seed = np.array([1., 0., 0.]) if abs(n[0]) < .9 else np.array([0., 1., 0.])
    u = seed - n * np.dot(seed, n)
    u /= np.linalg.norm(u)
    return np.column_stack([u, np.cross(n, u), n])
