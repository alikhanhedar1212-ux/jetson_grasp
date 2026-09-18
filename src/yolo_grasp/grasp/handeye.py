"""Offline Eye-in-Hand geometry. T_A_B maps B coordinates into A; metres."""
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import transform, apply, pose_error

BOARD = {
    'inner_corners': [9, 6],
    # Measured physical corner-to-corner spacings, starting at the operator-
    # confirmed origin beside the printed "www" edge.  The print alternates
    # 20.5/20.0 mm rather than forming an exact uniform 20 mm grid.
    'x_spacings_m': [.0205, .0200, .0205, .0200, .0205, .0200, .0205, .0200],
    'y_spacings_m': [.0205, .0200, .0205, .0200, .0205],
}
SERIAL = '260322275595'


def board_points():
    points = np.zeros((54, 3), np.float64)
    x = np.r_[0., np.cumsum(BOARD['x_spacings_m'])]
    y = np.r_[0., np.cumsum(BOARD['y_spacings_m'])]
    points[:, :2] = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 2)
    return points


def intrinsics_matrix(intr):
    k = np.array([[intr['fx'], 0, intr['ppx']], [0, intr['fy'], intr['ppy']], [0, 0, 1.]])
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid camera intrinsics')
    return k


def normalized_corners(corners, intr):
    k = intrinsics_matrix(intr)
    model = intr['model'].split('.')[-1]
    if model == 'none':
        return cv2.undistortPoints(np.asarray(corners, np.float64), k, np.zeros(5)).reshape(-1, 2)
    if model == 'brown_conrady':
        return cv2.undistortPoints(np.asarray(corners, np.float64), k, np.asarray(intr['coeffs'])).reshape(-1, 2)
    if model == 'inverse_brown_conrady':
        # librealsense implements inverse Brown deprojection. Persist the rays
        # so offline solve has no dependency on a connected RealSense.
        import pyrealsense2 as rs
        i = rs.intrinsics()
        for key in ('width', 'height', 'fx', 'fy', 'ppx', 'ppy', 'coeffs'):
            setattr(i, key, intr[key])
        i.model = rs.distortion.inverse_brown_conrady
        rays = np.array([rs.rs2_deproject_pixel_to_point(i, p.tolist(), 1.) for p in np.asarray(corners).reshape(-1, 2)])
        return rays[:, :2] / rays[:, 2:]
    raise ValueError(f'Unsupported color distortion {model}; do not treat it as zero distortion')


def detect_board(bgr, flip=False):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if hasattr(cv2, 'findChessboardCornersSB'):
        ok, corners = cv2.findChessboardCornersSB(gray, (9, 6), cv2.CALIB_CB_NORMALIZE_IMAGE)
    else:
        ok, corners = cv2.findChessboardCorners(gray, (9, 6))
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 40, .001))
    if not ok:
        raise ValueError('Need all 9 x 6 inner corners visible')
    corners = corners.reshape(-1, 2).astype(np.float64)
    if flip:
        corners = corners[::-1].copy()
    return corners


def estimate_board(corners, intr, rays=None):
    corners = np.asarray(corners, np.float64).reshape(-1, 2)
    if corners.shape != (54, 2) or not np.isfinite(corners).all():
        raise ValueError('Expected 54 finite corners')
    rays = normalized_corners(corners, intr) if rays is None else np.asarray(rays, np.float64)
    if rays.shape != (54, 2) or not np.isfinite(rays).all():
        raise ValueError('Invalid normalized corner rays')
    obj = board_points()
    ok, rv, tv = cv2.solvePnP(obj, rays, np.eye(3), None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError('PnP failed')
    t = np.eye(4)
    t[:3, :3] = cv2.Rodrigues(rv)[0]
    t[:3, 3] = tv.ravel()
    camera = apply(t, obj)
    if np.min(camera[:, 2]) <= 0:
        raise ValueError('PnP board behind camera')
    # Pixel-equivalent residual in undistorted image coordinates; this is not
    # a raw distorted-image residual for nonzero distortion.
    error = (camera[:, :2] / camera[:, 2:] - rays) * [intr['fx'], intr['fy']]
    rms = float(np.sqrt(np.mean(np.sum(error ** 2, axis=1))))
    if rms > 1.:
        raise ValueError(f'PnP undistorted pixel RMS {rms:.3f} > 1 px')
    return transform(t), rms, rays


def mean_transform(values):
    t = np.eye(4)
    t[:3, :3] = Rotation.from_matrix(np.array([v[:3, :3] for v in values])).mean().as_matrix()
    t[:3, 3] = np.mean([v[:3, 3] for v in values], axis=0)
    return t


def diversity(poses):
    vectors = np.array([Rotation.from_matrix(a[:3, :3].T @ b[:3, :3]).as_rotvec()
                        for i, a in enumerate(poses) for b in poses[i+1:]])
    if len(vectors) < 3:
        raise ValueError('Insufficient poses')
    singular = np.linalg.svd(vectors, compute_uv=False)
    max_angle = float(np.rad2deg(np.max(np.linalg.norm(vectors, axis=1))))
    if singular[0] < 1e-8 or singular[1] / singular[0] < .1 or max_angle < 15:
        raise ValueError('Degenerate rotations: use >=15 degree diversity around at least two nonparallel axes')
    for i, a in enumerate(poses):
        if any(pose_error(a, b)[0] < .003 and pose_error(a, b)[1] < np.deg2rad(2) for b in poses[:i]):
            raise ValueError('Near duplicate poses; recollect genuinely different views')
    return {'rotation_singular_values': singular.tolist(), 'max_pair_rotation_deg': max_angle}


def residuals(samples, x, reference):
    rows = []
    for sample in samples:
        b = transform(sample['T_base_flange']) @ x @ transform(sample['T_camera_board'])
        p, r = pose_error(reference, b)
        corner_err = np.linalg.norm(apply(b, board_points()) - apply(reference, board_points()), axis=1)
        rows.append({'id': sample['id'], 'origin_error_mm': p * 1000,
                     'rotation_error_deg': float(np.rad2deg(r)),
                     'corner_rms_mm': float(np.sqrt(np.mean(corner_err ** 2)) * 1000)})
    return rows


def fit(samples, method):
    a = [transform(s['T_base_flange']) for s in samples]
    b = [transform(s['T_camera_board']) for s in samples]
    r, t = cv2.calibrateHandEye([v[:3, :3] for v in a], [v[:3, 3] for v in a],
        [v[:3, :3] for v in b], [v[:3, 3] for v in b], method=method)
    x = np.eye(4)
    x[:3, :3], x[:3, 3] = r, np.asarray(t).ravel()
    transform(x)
    reference = mean_transform([aa @ x @ bb for aa, bb in zip(a, b)])
    return x, reference


def solve(samples, max_mm=5., max_deg=2., tcp=None):
    train = [s for s in samples if s['split'] == 'train']
    test = [s for s in samples if s['split'] == 'validation']
    if len(train) < 15 or len(test) < 5:
        raise ValueError('Need >=15 training and >=5 separately captured validation poses')
    coverage = diversity([transform(s['T_base_flange']) for s in train])
    diversity([transform(s['T_base_flange']) for s in samples])
    candidates = {}
    models = {}
    for name in ('PARK', 'TSAI', 'HORAUD'):
        try:
            x, reference = fit(train, getattr(cv2, 'CALIB_HAND_EYE_' + name))
            rows = residuals(train, x, reference)
            score = float(np.mean([s['corner_rms_mm'] for s in rows]))
            candidates[name] = {'train_mean_corner_mm': score, 'T_flange_wrist': x.tolist()}
            models[name] = (score, x, reference, rows)
        except (ValueError, cv2.error, np.linalg.LinAlgError) as error:
            candidates[name] = {'error': str(error)}
    if not models:
        raise ValueError('All hand-eye methods failed')
    name = min(models, key=lambda n: models[n][0])
    _, x, reference, rows = models[name]
    heldout = residuals(test, x, reference)
    passed = all(r['corner_rms_mm'] <= max_mm and r['rotation_error_deg'] <= max_deg for r in rows + heldout)
    result = {'schema_version': 1, 'validated': False, 'numerical_checks_passed': bool(passed),
        'convention': 'T_A_B maps B into A; metres; color optical camera',
        'camera_serial': SERIAL, 'board': BOARD, 'method': name, 'opencv_version': cv2.__version__,
        'T_flange_wrist': x.tolist(), 'T_base_board': reference.tolist(),
        'T_flange_tcp': None, 'T_tcp_camera': None,
        'thresholds': {'max_corner_rms_mm': max_mm, 'max_rotation_deg': max_deg},
        'diversity': coverage, 'candidates': candidates, 'train_residuals': rows, 'validation_residuals': heldout,
        'note': 'Numerical consistency is not independent physical accuracy. Do not enable grasping from this file alone.'}
    if tcp is not None:
        tcp = transform(tcp)
        result.update(T_flange_tcp=tcp.tolist(), T_tcp_camera=(np.linalg.inv(tcp) @ x).tolist())
    return result


def load_samples(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'session.json').read_text())
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported session schema')
    if manifest['camera_serial'] != SERIAL or manifest['board'] != BOARD or manifest['robot_frame'] != 'flange':
        raise ValueError('Unexpected camera/board/robot-frame identity')
    samples = []
    intrinsics = None
    for path in sorted(directory.glob('sample_*/sample.json')):
        s = json.loads(path.read_text())
        if s.get('schema_version') != 1 or not s.get('origin_operator_confirmed'):
            raise ValueError(f'Unconfirmed board origin/schema: {path}')
        if s['split'] not in ('train', 'validation') or s['camera_serial'] != SERIAL:
            raise ValueError(f'Invalid sample identity: {path}')
        intr = s['color_intrinsics']
        if intrinsics is not None and intr != intrinsics:
            raise ValueError('Stream intrinsics changed within session')
        intrinsics = intr
        t, rms, _ = estimate_board(s['corners_px'], intr, s['normalized_corners'])
        s['T_camera_board'], s['pnp_rms_px'], s['id'] = t.tolist(), rms, path.parent.name
        transform(s['T_base_flange'])
        samples.append(s)
    return manifest, samples


def camera_to_base(point, base_flange, result, serial=SERIAL):
    if result['camera_serial'] != serial:
        raise ValueError('Camera serial mismatch')
    if not result.get('numerical_checks_passed'):
        raise ValueError('Hand-eye numerical checks did not pass')
    point = np.asarray(point, float)
    if point.shape != (3,) or not np.isfinite(point).all() or point[2] <= 0:
        raise ValueError('Expected finite camera optical XYZ in metres with positive depth')
    return apply(transform(base_flange) @ transform(result['T_flange_wrist']), point)
