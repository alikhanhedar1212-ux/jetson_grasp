"""Detect the black frame lying on the blue mat: classical CV, no model.

The placement flow needs the black frame's position, not the red block's, and
there is no trained model for it. Segmentation is therefore explicit: find the
blue mat, take the dark (low value, low saturation) pixels that lie on it, and
report only interior candidates, preferring the upper frame. Shadows on the mat are dark blue
(high saturation) and are rejected by the saturation test.
"""
import numpy as np


def find_black_frame(bgr, *, dark_v=70, dark_s=90, blue_h=(90, 140), blue_s=60,
                     blue_v=40, min_area_px=200, max_area_frac=.6):
    """Return the uppermost black frame candidate inside the blue mat."""
    found = find_black_frames(bgr, dark_v=dark_v, dark_s=dark_s, blue_h=blue_h, blue_s=blue_s,
                              blue_v=blue_v, min_area_px=min_area_px, max_area_frac=max_area_frac)
    if found['candidates']:
        best = dict(found['candidates'][0])
        best.update(valid=True, reason=None, blue_fraction=found['blue_fraction'],
                    dark_fraction=found['dark_fraction'])
        return best
    return dict(valid=False, reason=found['reason'], box_xyxy=None, area_px=0., center_uv=None,
                blue_fraction=found['blue_fraction'], dark_fraction=found['dark_fraction'])


def find_black_frames(bgr, *, dark_v=70, dark_s=90, blue_h=(90, 140), blue_s=60,
                      blue_v=40, min_area_px=200, max_area_frac=.6):
    """All interior black-on-blue candidates, uppermost first.

    ``reason`` is set (``no_blue_mat``, ``no_black_frame``,
    ``black_frame_too_small``, ``black_frame_too_large``) when the list is empty;
    the caller keeps the same reject-and-retry policy as the red block.
    """
    import cv2
    bgr = np.asarray(bgr)
    if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
        raise ValueError('需要 uint8 BGR 图像')
    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    result = dict(candidates=[], reason=None, blue_fraction=0., dark_fraction=0.)
    blue = ((hue >= blue_h[0]) & (hue <= blue_h[1]) & (sat >= blue_s) & (val >= blue_v)).astype(np.uint8)
    result['blue_fraction'] = float(blue.mean())
    if result['blue_fraction'] < .02:
        result['reason'] = 'no_blue_mat'
        return result
    # Fill only the outer contour of the largest connected blue board. Holes
    # inside it may be black frames; background outside it is never a candidate.
    board_blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    boards, _ = cv2.findContours(board_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    board = np.zeros_like(blue)
    cv2.drawContours(board, [max(boards, key=cv2.contourArea)], -1, 1, cv2.FILLED)
    interior = cv2.erode(board, np.ones((5, 5), np.uint8))
    dark = (((val <= dark_v) & (sat <= dark_s)) & (board != 0)).astype(np.uint8)
    result['dark_fraction'] = float(dark.mean())
    mask = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    mask &= board
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        result['reason'] = 'no_black_frame'
        return result
    pad = 6
    rejected = None
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = float(cv2.contourArea(contour))
        x, y, box_w, box_h = cv2.boundingRect(contour)
        if area < min_area_px:
            rejected = rejected or 'black_frame_too_small'
            continue
        if float(box_w * box_h) > max_area_frac * h * w:
            rejected = 'black_frame_too_large'
            continue
        # Reject clipped background/gripper fragments at the board boundary.
        filled = np.zeros_like(board)
        cv2.drawContours(filled, [contour], -1, 1, cv2.FILLED)
        if np.any((filled != 0) & (interior == 0)):
            continue
        # The frame must lie on the mat: the band around its bounding box has to
        # be blue. A filled black box has no blue inside, so the test is on the ring.
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + box_w + pad), min(h, y + box_h + pad)
        ring = np.zeros_like(blue)
        ring[y0:y1, x0:x1] = 1
        ring[y:y + box_h, x:x + box_w] = 0
        around = float(blue[ring == 1].mean()) if ring.sum() else 0.
        if around < .3:
            continue
        result['candidates'].append(dict(
            box_xyxy=[float(x), float(y), float(x + box_w), float(y + box_h)],
            area_px=area, blue_around=around,
            center_uv=[int(x + box_w // 2), int(y + box_h // 2)]))
    if not result['candidates']:
        result['reason'] = rejected or 'no_black_frame'
    result['candidates'].sort(key=lambda c: (c['center_uv'][1], -c['area_px']))
    return result


class _Scalar:
    def __init__(self, value):
        self.value = value
    def item(self):
        return self.value


class _Tensor:
    """Tiny stand-in for a torch tensor: the callers use .cpu().numpy()."""
    def __init__(self, array):
        self.array = np.asarray(array, float)
    def cpu(self):
        return self
    def numpy(self):
        return self.array


class _Box:
    def __init__(self, xyxy, confidence=1., cls=0):
        self.xyxy = [_Tensor(xyxy)]
        self.conf = _Scalar(float(confidence))
        self.cls = _Scalar(int(cls))


class _Prediction:
    """Prediction-shaped result so the existing preview/observation code works."""
    def __init__(self, boxes, reason, image, cv2):
        self.boxes = boxes
        self.reason = reason
        self.image = None if image is None else np.array(image, copy=True)
        self.cv2 = cv2
        self.names = BlackFrameModel.names

    def plot(self):
        if self.image is None or self.cv2 is None:
            return self.image
        canvas = self.image
        for box in self.boxes:
            x1, y1, x2, y2 = np.rint(box.xyxy[0].array).astype(int)
            self.cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.cv2.putText(canvas, 'black_frame', (max(0, x1), max(16, y1 - 6)),
                             self.cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
        if not self.boxes:
            self.cv2.putText(canvas, f'no black frame ({self.reason})', (10, 30),
                             self.cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 165, 255), 2)
        return canvas


class BlackFrameModel:
    """Model-shaped adapter: same .predict(source=...) contract as YOLO.

    Every candidate is returned as a box; the depth- and size-based choice
    happens in :func:`black_frame_point`, which also has the RGB-D frame.
    """
    names = {0: 'black_frame'}

    def __init__(self, finder=find_black_frames, cv2=None):
        self.finder = finder
        self.cv2 = cv2

    def predict(self, source=None, **kwargs):
        found = self.finder(source)
        boxes = [_Box(candidate['box_xyxy']) for candidate in found.get('candidates', [])]
        return [_Prediction(boxes, found.get('reason'), source, self.cv2)]


def select_black_frame(boxes_or_candidates, frame, size_m=(.070, .060), tolerance=.30,
                       min_depth=.10, max_depth=.60):
    """Prefer the upper candidate among depth- and physical-size-valid frames.

    The mat carries more than one dark region (the gripper itself shows up in
    front of the mat at ~80 mm, and the lower rectangle can be occluded), so area
    alone picks the wrong one. The known size (70 x 60 mm by default) at the
    candidate's own depth is used instead.
    """
    depth = np.asarray(frame.depth_m, float)
    metadata = getattr(frame, 'metadata', None) or {}
    intrinsics = metadata.get('intrinsics') or metadata.get('color_intrinsics') or {}
    fx, fy = float(intrinsics.get('fx', 0.)), float(intrinsics.get('fy', 0.))
    if fx <= 0 or fy <= 0:
        # No intrinsics: keep the detector priority and report no size check.
        first = boxes_or_candidates[0] if boxes_or_candidates else None
        if first is None:
            return None
        box = first['box_xyxy'] if isinstance(first, dict) else \
            np.asarray(first.xyxy[0].cpu().numpy(), float)
        x1, y1, x2, y2 = [float(v) for v in box]
        return dict(index=0, box_xyxy=[x1, y1, x2, y2],
                    center_uv=[int((x1+x2)//2), int((y1+y2)//2)],
                    depth_m=None, expected_px=None, size_error=None,
                    selection='upper first（帧缺少焦距，未做尺寸核对）')
    best = None
    for index, item in enumerate(boxes_or_candidates):
        box = item['box_xyxy'] if isinstance(item, dict) else np.asarray(item.xyxy[0].cpu().numpy(), float)
        x1, y1, x2, y2 = [float(v) for v in box]
        u, v = int((x1+x2)//2), int((y1+y2)//2)
        if not (0 <= v < depth.shape[0] and 0 <= u < depth.shape[1]):
            continue
        z = float(depth[v, u])
        if not np.isfinite(z) or not min_depth <= z <= max_depth:
            continue
        expected_w, expected_h = fx*size_m[0]/z, fy*size_m[1]/z
        error = max(abs((x2-x1)-expected_w)/expected_w, abs((y2-y1)-expected_h)/expected_h)
        record = dict(index=index, box_xyxy=[x1, y1, x2, y2], center_uv=[u, v], depth_m=z,
                      expected_px=[expected_w, expected_h], size_error=error)
        if error <= tolerance and (best is None or
                (v, error) < (best['center_uv'][1], best['size_error'])):
            best = record
    if best is not None:
        best['selection'] = 'size match; upper frame first'
    return best


def black_frame_point(prediction, frame, locate, min_depth=.07, max_depth=.6,
                      size_m=(.070, .060), tolerance=.30):
    """Same shape as the red-block point: depth at the chosen box centre plus checks."""
    boxes = getattr(prediction, 'boxes', [])
    if not boxes:
        point = dict(valid=False, reason='black_frame_count', count=0)
        detector_reason = getattr(prediction, 'reason', None)
        if detector_reason is not None:
            point['detector_reason'] = detector_reason
        return point
    chosen = select_black_frame(boxes, frame, size_m=size_m, tolerance=tolerance)
    if chosen is None:
        return dict(valid=False, reason='black_frame_size_mismatch', count=len(boxes),
                    detail='没有候选的像素尺寸与已知矩形尺寸相符（可能夹爪/遮挡物被当成黑框）')
    point = dict(locate(frame, chosen['box_xyxy'],
                        min_depth=min_depth, max_depth=max_depth))
    point['frame_candidate'] = chosen
    if not point['valid'] and point.get('pixel_uv') is not None:
        u, v = point['pixel_uv']
        if 0 <= v < frame.depth_m.shape[0] and 0 <= u < frame.depth_m.shape[1]:
            raw = float(frame.depth_m[v, u])
            point['center_raw_m'] = raw if np.isfinite(raw) else None
            point['center_depth_category'] = (
                'nonfinite' if not np.isfinite(raw) else 'zero_or_negative' if raw <= 0
                else 'below_70mm' if raw < .07 else 'above_600mm' if raw > .6 else 'in_range')
    return point
