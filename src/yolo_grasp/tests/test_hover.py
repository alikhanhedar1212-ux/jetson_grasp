import ast
from pathlib import Path
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from grasp.hover_test import TCP_OFFSET_M,hover_target,solve_target,main


def test_transform_direction_units_tcp_orientation():
    flange=np.eye(4);flange[:3,:3]=Rotation.from_euler('z',90,degrees=True).as_matrix();flange[:3,3]=[.1,.2,.3]
    handeye=np.eye(4);handeye[:3,3]=[.02,0,0]
    rotation=Rotation.from_euler('x',180,degrees=True).as_matrix()
    target,point=hover_target(flange,[.03,.04,.2],handeye,rotation,.125)
    np.testing.assert_allclose(point,[.06,.25,.5])
    np.testing.assert_allclose(target[:3,:3],rotation)
    tcp=np.eye(4);tcp[:3,3]=TCP_OFFSET_M
    np.testing.assert_allclose((target@tcp)[:3,3],[.06,.25,.625])
    with pytest.raises(ValueError):hover_target(flange,[0,0,.2],handeye,clearance=.005)


def test_unreachable_rejected():
    target=np.eye(4);target[0,3]=.05
    with pytest.raises(ValueError):solve_target(target,np.zeros(6),[[-1,1]]*6,lambda q:np.eye(4))


def test_tool_yaw_prefers_nearest_feasible_attitude(monkeypatch):
    import grasp.hover_test as m
    attempted = []
    def fake_diagnose(*args, orientation_override=None, **kwargs):
        yaw = 0 if orientation_override is None else round(
            Rotation.from_matrix(orientation_override).as_euler('zyx', degrees=True)[0])
        attempted.append(yaw)
        # +5° has an IK but violates the fixed-pose 90° envelope; +10° is the
        # first fully acceptable solution and must win over every larger angle.
        fixed_delta = 91 if yaw == 5 else 80
        valid = yaw in (5, 10)
        return dict(IK_joints_rad=[0]*6 if valid else None,
                    IK_delta_from_current_deg=[20]*6,
                    IK_delta_from_fixed_deg=[fixed_delta]*6,
                    issues=[])
    monkeypatch.setattr(m, 'diagnose', fake_diagnose)
    args = (np.zeros(6), np.eye(4), [0, 0, .2], np.eye(4), [], lambda q: np.eye(4),
            .035, 90.)
    report = m.diagnose_best_tool_yaw(*args)
    assert attempted == [0, 5, -5, 10]
    assert report['tool_yaw_from_fixed_deg'] == 10
    assert report['tool_yaw_search_deg'] == [-45., 45.]


def test_tool_yaw_rejects_instead_of_searching_beyond_45_degrees(monkeypatch):
    import grasp.hover_test as m
    attempted = []
    def fake_diagnose(*args, orientation_override=None, **kwargs):
        yaw = 0 if orientation_override is None else round(
            Rotation.from_matrix(orientation_override).as_euler('zyx', degrees=True)[0])
        attempted.append(yaw)
        return dict(IK_joints_rad=None, issues=['fixed failure'])
    monkeypatch.setattr(m, 'diagnose', fake_diagnose)
    args = (np.zeros(6), np.eye(4), [0, 0, .2], np.eye(4), [], lambda q: np.eye(4),
            .035, 90.)
    report = m.diagnose_best_tool_yaw(*args)
    assert min(attempted) == -45 and max(attempted) == 45 and len(attempted) == 19
    assert report['tool_yaw_from_fixed_deg'] is None
    assert any('±45°范围内无满足' in issue for issue in report['issues'])


def test_execution_rejected_and_no_actuator_api():
    with pytest.raises(SystemExit) as error:main(['--execute'])
    assert error.value.code==2
    tree=ast.parse((Path(__file__).resolve().parents[1]/'grasp/hover_test.py').read_text())
    forbidden={'move_j','move_p','move_l','enable','disable','reset','electronic_emergency_stop','create_arm','command'}
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in forbidden]


def test_depth_diagnostic_distinguishes_hole_and_range():
    from types import SimpleNamespace
    from grasp.hover_test import depth_diagnostic
    depth=np.full((21,21),.2)
    frame=SimpleNamespace(depth_m=depth,metadata={'depth_scale_m':.00001})
    for value,category in [(0,'zero_or_negative'),(.05,'below_70mm'),(.7,'above_600mm'),(float('nan'),'nonfinite')]:
        depth[10,10]=value
        report=depth_diagnostic(frame,[3,3,17,17],[10,10])
        assert report['center_category']==category
        assert report['neighborhood_11x11']['positive_median_mm']==200
    assert np.isnan(depth[10,10])


def test_ten_mm_clearance_is_tcp_base_z_not_flange_z():
    flange = np.eye(4)
    flange[:3, :3] = Rotation.from_euler('y', 130, degrees=True).as_matrix()
    flange[:3, 3] = [.2, 0, .35]
    target, base = hover_target(flange, [0, 0, .2], np.eye(4), clearance=.010)
    tcp = np.eye(4); tcp[:3, 3] = TCP_OFFSET_M
    np.testing.assert_allclose((target @ tcp)[:3, 3]-base, [0, 0, .010], atol=1e-15)
    assert abs(target[2, 3]-base[2]-.010) > .01
    with pytest.raises(ValueError):
        hover_target(flange, [0, 0, .2], np.eye(4), clearance=.009)


def test_travel_extension_requires_explicit_parameter(monkeypatch):
    import grasp.hover_test as m
    target = np.eye(4); target[0, 3] = .151
    monkeypatch.setattr(m, 'hover_target', lambda *args: (target, np.zeros(3)))
    monkeypatch.setattr(m, 'solve_target', lambda target, joints, *args: joints)
    args = (np.deg2rad(m.TARGET_DEG), np.eye(4), [0, 0, .2], np.eye(4), [], lambda q: np.eye(4))
    default = m.diagnose(*args)
    assert '目标法兰位移超过150 mm' in default['issues']
    extended = m.diagnose(*args, max_travel_mm=200, max_delta_deg=90)
    assert extended['issues'] == []
    assert extended['max_flange_travel_mm'] == 200
    assert extended['max_hover_joint_delta_deg'] == 90
    with pytest.raises(ValueError): m.diagnose(*args, max_travel_mm=201)


def test_target_offset_trims_the_tcp_in_the_base_frame():
    flange = np.eye(4); flange[:3, 3] = [.2, 0, .35]
    plain, _ = hover_target(flange, [0, 0, .2], np.eye(4), clearance=.010)
    shift = np.array([-.005, 0, 0])
    trimmed, _ = hover_target(flange, [0, 0, .2], np.eye(4), clearance=.010,
                              target_offset_m=shift)
    tcp = np.eye(4); tcp[:3, 3] = TCP_OFFSET_M
    np.testing.assert_allclose((trimmed @ tcp)[:3, 3] - (plain @ tcp)[:3, 3], shift)
    with pytest.raises(ValueError): hover_target(flange, [0, 0, .2], np.eye(4),
                                                 target_offset_m=[0, 0, .2])


def test_diagnose_records_the_target_trim(monkeypatch):
    import grasp.hover_test as m
    target = np.eye(4); target[0, 3] = .151
    monkeypatch.setattr(m, 'hover_target', lambda *args, **kwargs: (target, np.zeros(3)))
    monkeypatch.setattr(m, 'solve_target', lambda target, joints, *args, **kwargs: joints)
    args = (np.deg2rad(m.TARGET_DEG), np.eye(4), [0, 0, .2], np.eye(4), [], lambda q: np.eye(4))
    report = m.diagnose(*args, max_travel_mm=200, max_delta_deg=90,
                        target_offset_m=(-.005, 0, 0))
    assert report['target_offset_mm'] == [-5.0, 0.0, 0.0]
    assert any('target trim' in line for line in report['assumptions'])


def test_tcp_lateral_offset_shifts_the_grasp_point_in_the_flange_frame():
    flange = np.eye(4)
    flange[:3, :3] = Rotation.from_euler('y', 40, degrees=True).as_matrix()
    flange[:3, 3] = [.2, 0, .35]
    plain, _ = hover_target(flange, [0, 0, .2], np.eye(4), clearance=.010)
    lateral = np.array([.004, -.003, .080])
    shifted, _ = hover_target(flange, [0, 0, .2], np.eye(4), clearance=.010,
                              tcp_offset_m=lateral)
    plain_tcp = np.eye(4); plain_tcp[:3, 3] = TCP_OFFSET_M
    shifted_tcp = np.eye(4); shifted_tcp[:3, 3] = lateral
    # Both flange targets still put their own TCP exactly 10 mm above the point.
    base = (flange @ np.r_[[0, 0, .2], 1])[:3]
    np.testing.assert_allclose((plain @ plain_tcp)[:3, 3]-base, [0, 0, .010], atol=1e-15)
    np.testing.assert_allclose((shifted @ shifted_tcp)[:3, 3]-base, [0, 0, .010], atol=1e-15)
    # The flange itself moved by the lateral part expressed in the base frame.
    expected = flange[:3, :3] @ (TCP_OFFSET_M-lateral)
    np.testing.assert_allclose(shifted[:3, 3]-plain[:3, 3], expected, atol=1e-15)
    assert abs(expected[0]) > .003 and abs(expected[2]) > .001
    with pytest.raises(ValueError): hover_target(flange, [0, 0, .2], np.eye(4),
                                                 tcp_offset_m=[.06, 0, .08])
    with pytest.raises(ValueError): hover_target(flange, [0, 0, .2], np.eye(4),
                                                 tcp_offset_m=[0, 0, .3])


def test_diagnose_records_the_tcp_offset(monkeypatch):
    import grasp.hover_test as m
    target = np.eye(4); target[0, 3] = .151
    monkeypatch.setattr(m, 'hover_target', lambda *args, **kwargs: (target, np.zeros(3)))
    monkeypatch.setattr(m, 'solve_target', lambda target, joints, *args, **kwargs: joints)
    args = (np.deg2rad(m.TARGET_DEG), np.eye(4), [0, 0, .2], np.eye(4), [], lambda q: np.eye(4))
    report = m.diagnose(*args, max_travel_mm=200, max_delta_deg=90,
                        tcp_offset_m=(.004, -.003, .078))
    assert report['tcp_offset_mm'] == [4.0, -3.0, 78.0]
    assert 'flange->TCP offset' in report['assumptions'][1]
