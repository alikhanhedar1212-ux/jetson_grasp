import json
import numpy as np
import pytest
from grasp.j6_camera import ordered_corners, Observation
from grasp.handeye import SERIAL, BOARD


def test_corner_reversal_and_ambiguity():
    corners = np.array([[x*20., y*20.] for y in range(6) for x in range(9)])
    ordered, flip = ordered_corners((corners + 2)[::-1], corners)
    assert flip
    np.testing.assert_allclose(ordered, corners + 2)
    with pytest.raises(ValueError):
        ordered_corners(corners + 200, corners)


def test_transform_chain_and_raw_save(tmp_path, monkeypatch):
    from grasp import j6_camera as mod
    x = np.eye(4); x[0,3] = .1
    path = tmp_path / 'calibration.json'
    path.write_text(json.dumps(dict(camera_serial=SERIAL, board=BOARD, T_flange_wrist=x.tolist(), validated=False)))
    obs = Observation(tmp_path, path)
    base = np.eye(4); base[1,3] = .2
    board = np.eye(4); board[2,3] = .3
    corners = np.array([[x*20., y*20.] for y in range(6) for x in range(9)])
    from types import SimpleNamespace
    class Frame:
        bgr = np.zeros((240,320,3),np.uint8)
        metadata = {'color_intrinsics': {}}
        def save(self, directory):
            directory.mkdir()
    obs.camera = SimpleNamespace(capture=lambda **kw: Frame())
    robot = dict(T_base_flange=base.tolist(), joints_rad=[0]*6, enabled=[True]*6, arrival_flag=0)
    obs.history = SimpleNamespace(window=lambda *a: (robot,[robot]))
    monkeypatch.setattr(mod.time, 'sleep', lambda t: None)
    monkeypatch.setattr(mod, 'detect_board', lambda b: corners)
    monkeypatch.setattr(mod, 'estimate_board', lambda *a: (board,.1,corners))
    row = obs._collect({'index':0,'offset_deg':0}, np.zeros(6))
    np.testing.assert_allclose(row['T_base_board'], base @ x @ board)
    assert row['valid'] and (tmp_path/'sample_0000/observation.json').exists()
    monkeypatch.setattr(mod, 'detect_board', lambda b: (_ for _ in ()).throw(ValueError('no board')))
    row = obs._collect({'index':1,'offset_deg':5}, np.zeros(6))
    assert not row['valid'] and row['error'] == 'no board'
