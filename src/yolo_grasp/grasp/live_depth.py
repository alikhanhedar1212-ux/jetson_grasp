"""D435i detection to aligned depth and optical-camera XYZ; no robot commands."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
from .camera import RealSense


def locate(frame, box, min_depth=.1, max_depth=2., radius=2, max_spread=.01):
    """Use the rounded box-centre pixel, rejecting holes and depth boundaries.

    XYZ is the SDK-deprojected point at that exact pixel, not an interpolated
    grasp centre. The neighbourhood only validates depth; it never fills holes.
    """
    result = dict(valid=False, reason=None, pixel_uv=None, depth_m=None, xyz_camera_m=None)
    box = np.asarray(box, dtype=float)
    h, w = frame.depth_m.shape
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        result['reason'] = 'invalid_box'
        return result
    u, v = np.floor((box[:2] + box[2:]) / 2 + .5).astype(int)
    result['pixel_uv'] = [int(u), int(v)]
    if not (0 <= u < w and 0 <= v < h):
        result['reason'] = 'center_out_of_bounds'
        return result
    z = float(frame.depth_m[v, u])
    if not np.isfinite(z) or not min_depth <= z <= max_depth:
        result['reason'] = 'invalid_center_depth'
        return result
    x1, x2 = max(0, u-radius), min(w, u+radius+1)
    y1, y2 = max(0, v-radius), min(h, v+radius+1)
    patch = frame.depth_m[y1:y2, x1:x2]
    good = np.isfinite(patch) & (patch >= min_depth) & (patch <= max_depth)
    if good.mean() < .7:
        result['reason'] = 'insufficient_neighbor_depth'
        return result
    if np.ptp(patch[good]) > max_spread:
        result['reason'] = 'depth_discontinuity'
        return result
    xyz = frame.points[v, u]
    if not np.isfinite(xyz).all() or xyz[2] <= 0 or abs(float(xyz[2])-z) > .002:
        result['reason'] = 'invalid_deprojection'
        return result
    result.update(valid=True, depth_m=z, xyz_camera_m=xyz.astype(float).tolist())
    return result


def surface_samples(valid_depth, valid_points, max_samples=192):
    """Evenly spaced sample of an ROI point cloud, near surface to far.

    The controller knows the camera pose and can therefore tell which surface
    is the block's *top*; this worker-side function only has to ship a compact
    representation of everything the ROI saw. A depth-ordered stride keeps the
    sample proportional to each surface's share of the box without depending on
    where in the image that surface happens to be.
    """
    order = np.argsort(valid_depth)
    if len(order) <= max_samples:
        return valid_points[order]
    index = np.linspace(0, len(order)-1, max_samples).astype(int)
    return valid_points[order[index]]


def locate_top_surface(frame, box, min_depth=.1, max_depth=2., inset_fraction=.12,
                       min_points=25, separation_m=.008):
    """Return the robust 3-D centre of the nearest surface inside a detection box.

    This is intended for the red grasp block viewed from above.  The inset keeps
    box-edge background and vertical sides out.  A significant depth gap splits
    the nearer top surface from farther geometry; otherwise the nearest 80% is
    retained.  The returned point is a robust point-cloud centre, not the depth
    at one image pixel.
    """
    result = dict(valid=False, reason=None, pixel_uv=None, depth_m=None,
                  xyz_camera_m=None, method='box_top_surface')
    box = np.asarray(box, dtype=float)
    h, w = frame.depth_m.shape
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        result['reason'] = 'invalid_box'
        return result
    if not (0 <= inset_fraction < .4 and isinstance(min_points, int) and min_points >= 9
            and np.isfinite(separation_m) and separation_m > 0):
        raise ValueError('顶面定位参数无效')
    width, height = box[2]-box[0], box[3]-box[1]
    x1 = max(0, int(np.ceil(box[0]+width*inset_fraction)))
    y1 = max(0, int(np.ceil(box[1]+height*inset_fraction)))
    x2 = min(w, int(np.floor(box[2]-width*inset_fraction))+1)
    y2 = min(h, int(np.floor(box[3]-height*inset_fraction))+1)
    result['roi_xyxy'] = [x1, y1, x2, y2]
    if x2 <= x1 or y2 <= y1:
        result['reason'] = 'top_surface_roi_empty'
        return result
    depth = frame.depth_m[y1:y2, x1:x2]
    points = frame.points[y1:y2, x1:x2]
    good = (np.isfinite(depth) & (depth >= min_depth) & (depth <= max_depth)
            & np.isfinite(points).all(axis=2) & (points[..., 2] > 0)
            & (np.abs(points[..., 2]-depth) <= .002))
    result['roi_pixel_count'] = int(good.size)
    result['valid_depth_count'] = int(good.sum())
    if good.sum() < min_points:
        result['reason'] = 'insufficient_top_surface_depth'
        return result
    valid_depth = depth[good]
    valid_points = points[good].astype(float)
    order = np.argsort(valid_depth)
    sorted_depth = valid_depth[order]
    lower, upper = max(1, int(.1*len(order))), min(len(order)-1, int(.9*len(order)))
    gaps = np.diff(sorted_depth)
    split = None
    if upper > lower:
        local = gaps[lower:upper]
        index = lower+int(np.argmax(local))
        if gaps[index] >= separation_m:
            split = index+1
    if split is None:
        threshold = float(np.quantile(valid_depth, .8))
        surface = valid_points[valid_depth <= threshold]
        selection = 'nearest_80_percent'
    else:
        surface = valid_points[order[:split]]
        threshold = float((sorted_depth[split-1]+sorted_depth[split])/2)
        selection = 'depth_gap'
    if len(surface) < min_points:
        result['reason'] = 'insufficient_top_surface_points'
        return result
    centre = np.median(surface, axis=0)
    distances = np.linalg.norm(surface-centre, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances-median_distance)))
    limit = median_distance+max(.002, 3*mad)
    inliers = surface[distances <= limit]
    if len(inliers) < min_points:
        result['reason'] = 'insufficient_top_surface_inliers'
        return result
    centre = np.mean(inliers, axis=0)
    all_uv = np.argwhere(good)
    surface_uv = all_uv[valid_depth <= threshold] if split is None else all_uv[order[:split]]
    # Pick the retained pixel nearest to the 3-D centre for display/diagnostics.
    chosen = int(np.argmin(np.linalg.norm(surface-centre, axis=1)))
    v, u = surface_uv[chosen]
    result.update(valid=True, pixel_uv=[int(u+x1), int(v+y1)],
                  depth_m=float(centre[2]), xyz_camera_m=centre.tolist(),
                  top_surface_point_count=int(len(surface)), inlier_count=int(len(inliers)),
                  depth_threshold_m=threshold, surface_selection=selection)
    samples = surface_samples(valid_depth, valid_points, max_samples=192)
    result['surface_samples_xyz_m'] = samples.tolist()
    result['surface_sample_count'] = int(len(samples))
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[2]/'models/red_block_best.pt')
    p.add_argument('--serial', default='243222074879')
    p.add_argument('--device', default='0')
    p.add_argument('--conf', type=float, default=.5)
    p.add_argument('--min-depth', type=float, default=.1, help='Minimum accepted depth in metres')
    p.add_argument('--max-depth', type=float, default=2.)
    p.add_argument('--max-spread', type=float, default=.01, help='Maximum 5x5 neighbourhood depth range, metres')
    p.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[2]/'data/depth_checks')
    args = p.parse_args(argv)
    if not args.model.is_file():
        p.error(f'Local model missing: {args.model}')
    if (not all(np.isfinite(v) for v in (args.conf, args.min_depth, args.max_depth, args.max_spread))
            or not 0 < args.conf <= 1 or not 0 < args.min_depth < args.max_depth or args.max_spread <= 0):
        p.error('Invalid confidence or depth thresholds')
    import cv2
    from ultralytics import YOLO
    model = YOLO(str(args.model), task='detect')
    class_ids = [int(k) for k, value in model.names.items() if value == 'red_block']
    if len(class_ids) != 1:
        raise ValueError('Model must contain exactly one red_block class')
    stream = dict(width=640, height=480, fps=30)
    cfg = dict(serial=args.serial, color=stream, depth=stream, warmup_frames=30,
               timeout_ms=5000, max_rgb_depth_skew_ms=35)
    window = 'D435i red_block Depth XYZ | S: save | Q/Esc: quit'
    batch_name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    saved = 0
    try:
        with RealSense(cfg) as camera:
            cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
            print('相机坐标 XYZ 单位 m：X 向右、Y 向下、Z 向前；S 保存，Q/Esc 退出。', flush=True)
            while True:
                start = time.monotonic()
                frame = camera.capture(discard=0)
                prediction = model.predict(source=frame.bgr, device=args.device, imgsz=640,
                                           conf=args.conf, classes=class_ids, verbose=False)[0]
                display = frame.bgr.copy()
                detections = []
                for i, b in enumerate(prediction.boxes):
                    box = b.xyxy[0].cpu().numpy()
                    point = locate(frame, box, args.min_depth, args.max_depth, max_spread=args.max_spread)
                    detections.append(dict(class_name='red_block', confidence=float(b.conf.item()),
                                           box_xyxy=box.astype(float).tolist(), **point))
                    x1, y1, x2, y2 = np.rint(box).astype(int)
                    color = (0, 255, 0) if point['valid'] else (0, 165, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(display, f'#{i+1} red_block {float(b.conf.item()):.2f}',
                                (max(0,x1), max(15,y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1)
                    if point['pixel_uv'] is not None:
                        u,v = point['pixel_uv']
                        if 0 <= u < 640 and 0 <= v < 480:
                            cv2.drawMarker(display, (u,v), color, cv2.MARKER_CROSS, 10, 1)
                fps = 1./max(time.monotonic()-start, 1e-9)
                # Separate readout panel prevents coordinates covering the target.
                panel = np.zeros((max(80, 36+len(detections)*48), 640, 3), dtype=np.uint8)
                cv2.putText(panel, f'FPS {fps:.1f} | camera XYZ (m) | targets {len(detections)} | saved {saved}',
                            (8,22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
                for i,d in enumerate(detections):
                    line = f"#{i+1} uv={d['pixel_uv']} "
                    if d['valid']:
                        x,y,z=d['xyz_camera_m'];line += f"Depth={d['depth_m']:.3f}m"
                        detail=f'X={x:+.4f}  Y={y:+.4f}  Z={z:+.4f} m'
                    else:
                        line += 'XYZ INVALID';detail=d['reason']
                    color=(0,255,0) if d['valid'] else (0,165,255)
                    for row,text in enumerate((line,detail)):
                        cv2.putText(panel,text,(8,44+i*48+row*20),cv2.FONT_HERSHEY_SIMPLEX,.48,color,1)
                if not detections:
                    cv2.putText(panel,'No red_block; no XYZ',(8,53),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,165,255),1)
                annotated = np.vstack((display,panel))
                cv2.imshow(window, annotated)
                key=cv2.waitKey(1)&0xff
                if key in (ord('q'),ord('Q'),27) or cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE)<1:
                    break
                if key in (ord('s'),ord('S')):
                    out=args.output/batch_name/f'{saved+1:06d}'
                    frame.metadata['observation'] = dict(
                        stage='live_box_center_depth', coordinate_frame='aligned_color_optical',
                        units='metres', model=str(args.model.resolve()), confidence_threshold=args.conf,
                        min_depth_m=args.min_depth,max_depth_m=args.max_depth,
                        max_neighbor_spread_m=args.max_spread,neighbor_radius_px=2,
                        detections=detections)
                    frame.save(out)
                    if not cv2.imwrite(str(out/'annotated.png'),annotated):
                        raise OSError('Could not save annotated image')
                    saved+=1
                    print(f'已保存: {out.resolve()}',flush=True)
    except KeyboardInterrupt:
        return 0
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
