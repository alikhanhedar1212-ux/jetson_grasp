import numpy as np
import pytest
from grasp.camera import Frame
from grasp.live_depth import locate, locate_top_surface


def frame():
    depth = np.full((20, 20), .5)
    points = np.zeros((20, 20, 3)); points[..., 2] = depth
    points[10, 10] = [.01, -.02, .5]
    return Frame(np.zeros((20,20,3), dtype=np.uint8),depth,points,{})


def test_exact_center_point_and_frame_roundtrip(tmp_path):
    f=frame();f.save(tmp_path/'frame');f=Frame.load(tmp_path/'frame')
    r=locate(f,[6,6,14,14])
    assert r['valid'] and r['pixel_uv']==[10,10]
    assert r['depth_m']==.5 and r['xyz_camera_m']==[.01,-.02,.5]


@pytest.mark.parametrize('value',[0.,float('nan'),float('inf'),3.])
def test_invalid_center_never_filled_from_neighbors(value):
    f=frame();f.depth_m[10,10]=value
    r=locate(f,[6,6,14,14])
    assert not r['valid'] and r['xyz_camera_m'] is None and r['depth_m'] is None


def test_depth_edge_rejected():
    f=frame();f.depth_m[8:13,11:13]=.54
    assert locate(f,[6,6,14,14])['reason']=='depth_discontinuity'


def test_sparse_neighborhood_rejected():
    f=frame();f.depth_m[8:10,8:13]=0
    assert locate(f,[6,6,14,14])['reason']=='insufficient_neighbor_depth'


def test_outside_and_bad_boxes():
    f=frame()
    assert locate(f,[-20,0,-2,10])['reason']=='center_out_of_bounds'
    assert locate(f,[8,8,2,2])['reason']=='invalid_box'


def test_bad_point_rejected():
    f=frame();f.points[10,10,0]=np.nan
    assert locate(f,[6,6,14,14])['reason']=='invalid_deprojection'


def test_top_surface_uses_box_point_cloud_not_center_pixel():
    h = w = 100
    depth = np.full((h, w), .60)
    points = np.zeros((h, w, 3), float)
    yy, xx = np.mgrid[:h, :w]
    points[..., 0] = (xx-50)*.001
    points[..., 1] = (yy-50)*.001
    points[..., 2] = depth
    # A top face surrounded by farther background/side pixels.  The exact
    # box centre is a hole, which must no longer invalidate the whole target.
    depth[25:75, 25:75] = .40
    points[25:75, 25:75, 2] = .40
    depth[50, 50] = 0
    points[50, 50] = np.nan
    f = Frame(np.zeros((h, w, 3), dtype=np.uint8), depth, points, {})
    result = locate_top_surface(f, [18, 18, 82, 82], min_depth=.07, max_depth=.8)
    assert result['valid'] and result['method'] == 'box_top_surface'
    assert result['surface_selection'] in ('depth_gap', 'nearest_80_percent')
    assert abs(result['depth_m']-.40) < 1e-12
    np.testing.assert_allclose(result['xyz_camera_m'][:2], [0., 0.], atol=1e-12)


def test_top_surface_rejects_sparse_or_invalid_roi():
    f = frame()
    f.depth_m[:] = 0
    f.points[:] = np.nan
    assert locate_top_surface(f, [2, 2, 18, 18])['reason'] == 'insufficient_top_surface_depth'
    assert locate_top_surface(f, [8, 8, 2, 2])['reason'] == 'invalid_box'


def test_top_surface_ships_a_compact_sample_of_the_whole_roi():
    """The controller needs points from both the top face and the side wall."""
    h = w = 100
    depth = np.full((h, w), .60)
    points = np.zeros((h, w, 3), float)
    yy, xx = np.mgrid[:h, :w]
    points[..., 0] = (xx-50)*.001
    points[..., 1] = (yy-50)*.001
    points[..., 2] = depth
    depth[20:70, 20:70] = .40
    points[20:70, 20:70, 2] = .40
    f = Frame(np.zeros((h, w, 3), dtype=np.uint8), depth, points, {})
    result = locate_top_surface(f, [10, 10, 90, 90], min_depth=.07, max_depth=.8)
    assert result['valid']
    samples = np.asarray(result['surface_samples_xyz_m'], float)
    assert result['surface_sample_count'] == len(samples) == 192
    assert np.unique(np.round(samples[:, 2], 3)).tolist() == [.40, .60]
    # Both surfaces keep roughly the share of the ROI they cover.
    counts = np.bincount(np.searchsorted([.5], samples[:, 2]))
    assert counts[0] > 25 and counts[1] > 25


def test_top_surface_sample_keeps_small_patches():
    result = locate_top_surface(frame(), [2, 2, 18, 18], min_depth=.07, max_depth=.8)
    assert result['valid'] and result['surface_sample_count'] == 169
