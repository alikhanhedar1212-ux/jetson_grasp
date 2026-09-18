import numpy as np
import pytest
from grasp.black_frame import BlackFrameModel, find_black_frame, find_black_frames


def scene(black_box=None, black_value=20, mat_value=(200, 120, 40), size=(240, 320)):
    """Blue mat (BGR) filling the frame, optionally with a black rectangle."""
    image = np.zeros((size[0], size[1], 3), np.uint8)
    image[:, :] = mat_value
    if black_box is not None:
        x1, y1, x2, y2 = black_box
        image[y1:y2, x1:x2] = black_value
    return image


def test_black_frame_on_the_blue_mat_is_found():
    found = find_black_frame(scene((100, 80, 200, 160)))
    assert found['valid'] and found['box_xyxy'] == [100, 80, 200, 160]
    assert found['center_uv'] == [150, 120]
    assert found['blue_fraction'] > .8 and found['blue_around'] > .9


def test_missing_mat_or_frame_is_reported():
    no_mat = np.zeros((60, 60, 3), np.uint8)
    no_mat[:, :] = 20                      # dark, not blue
    assert find_black_frame(no_mat)['reason'] == 'no_blue_mat'
    assert find_black_frame(scene(None))['reason'] == 'no_black_frame'


def test_small_dark_speck_is_rejected():
    found = find_black_frame(scene((10, 10, 18, 18)))
    assert not found['valid'] and found['reason'] == 'black_frame_too_small'


def test_dark_blue_shadow_on_the_mat_is_not_a_black_frame():
    image = scene(None, mat_value=(200, 120, 40))
    image[60:180, 60:180] = [140, 90, 40]      # dark blue: low value, high saturation
    assert find_black_frame(image)['reason'] == 'no_black_frame'


def test_prediction_adapter_matches_the_yolo_contract():
    model = BlackFrameModel()
    prediction = model.predict(source=scene((100, 80, 200, 160)))[0]
    assert model.names == {0: 'black_frame'}
    assert len(prediction.boxes) == 1
    box = prediction.boxes[0]
    np.testing.assert_allclose(box.xyxy[0].cpu().numpy(), [100, 80, 200, 160])
    assert box.cls.item() == 0 and box.conf.item() == pytest.approx(1.)
    assert prediction.plot().shape == (240, 320, 3)
    empty = model.predict(source=scene(None))[0]
    assert empty.boxes == [] and empty.reason == 'no_black_frame'


def test_two_rectangles_are_listed_largest_first_and_the_upper_one_can_win():
    image = scene((60, 40, 130, 100))          # 70 x 60 px, upper
    image[150:190, 80:120] = 20                # 40 x 40 px, lower (e.g. partly hidden)
    found = find_black_frames(image)
    assert len(found['candidates']) == 2
    assert found['candidates'][0]['box_xyxy'] == [60, 40, 130, 100]
    assert found['candidates'][0]['area_px'] > found['candidates'][1]['area_px']
    assert find_black_frame(image)['box_xyxy'] == [60, 40, 130, 100]


def test_black_background_and_board_edge_occluder_are_excluded():
    image = np.full((300, 400, 3), 20, np.uint8)
    image[30:280, 40:250] = (200, 120, 40)
    image[60:120, 90:160] = 20
    # A black gripper overlaps the edge and must not turn into a clipped box.
    image[170:240, :130] = 20
    found = find_black_frames(image)
    assert [c['box_xyxy'] for c in found['candidates']] == [[90., 60., 160., 120.]]


def test_upper_valid_frame_wins_over_larger_lower_frame():
    from grasp.black_frame import select_black_frame
    from types import SimpleNamespace
    image = scene((60, 30, 130, 90))
    image[130:196, 70:147] = 20
    candidates = find_black_frames(image)['candidates']
    assert len(candidates) == 2
    frame = SimpleNamespace(depth_m=np.full((240, 320), .4),
                            metadata={'intrinsics': {'fx': 400., 'fy': 400.}})
    best = select_black_frame(list(reversed(candidates)), frame)
    assert best['center_uv'] == [95, 60]
    # Upper preference never overrides invalid depth.
    frame.depth_m[60, 95] = 0.
    assert select_black_frame(candidates, frame)['center_uv'][1] == 163
