"""Receive-only coordinate audit. There is deliberately no motion execution path."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from .geometry import pose_matrix, matrix_pose, transform, pose_error
from .mat_pose import TARGET_DEG
from .piper_model import piper_planning_limits


MAX_HOVER_JOINT_DELTA_DEG = 30.0
# Flange -> grasp point offset, in the flange frame, metres: [x, y, z].
# The user measured the distance along flange-local +Z as 80 mm on 2026-09-11
# (60 mm drove the gripper below the block). The lateral x/y stay zero until a
# real tool measurement replaces them; --tcp-offset-mm overrides the whole
# vector per run so the residual can be trimmed without editing code.
TCP_OFFSET_M = np.array([0., 0., .080])


def hover_target(flange, camera_point, handeye, orientation=None, clearance=.125,
                 target_offset_m=(0., 0., 0.), tcp_offset_m=None):
    flange, handeye = transform(flange), transform(handeye)
    point = np.asarray(camera_point, float)
    if point.shape != (3,) or not np.isfinite(point).all() or point[2] <= 0:
        raise ValueError('Camera XYZ 必须为三个有限米值且 Z>0')
    tcp_offset_m = np.asarray(TCP_OFFSET_M if tcp_offset_m is None else tcp_offset_m, float)
    if (tcp_offset_m.shape != (3,) or not np.isfinite(tcp_offset_m).all()
            or np.max(np.abs(tcp_offset_m[:2])) > .05 or not 0 < tcp_offset_m[2] <= .2):
        raise ValueError('TCP 偏移必须为三个有限米值：横向每个不超过 ±50 mm，轴向 (0,200] mm')
    # Empirical trim in the base frame for the residual left by the uncalibrated
    # TCP direction and handeye; it shifts the target horizontally/vertically
    # without changing the approach orientation.
    offset = np.asarray(target_offset_m, float)
    if offset.shape != (3,) or not np.isfinite(offset).all() or np.max(np.abs(offset)) > .05:
        raise ValueError('目标修正量必须为三个有限米值，每个不超过 50 mm')
    # The user requested a 10 mm approach. Finger and table clearance
    # below 100 mm is not verified by any measurement.
    if not .010 <= clearance <= .150:
        raise ValueError('上方高度必须为 10–150 mm')
    base = (flange @ handeye @ np.r_[point, 1])[:3]
    desired_tcp = np.eye(4)
    desired_tcp[:3,:3] = flange[:3,:3] if orientation is None else orientation
    desired_tcp[:3,3] = base + [0, 0, clearance] + offset
    tcp = np.eye(4); tcp[:3,3] = tcp_offset_m
    return transform(desired_tcp @ np.linalg.inv(tcp)), base


def solve_target(target, start, limits, fk, max_delta_deg=MAX_HOVER_JOINT_DELTA_DEG):
    """Numerical IK with a bound on how far a single joint may leave the seed."""
    if not np.isfinite(max_delta_deg) or not 0 < max_delta_deg <= 180:
        raise ValueError('单关节变化上限必须在 (0,180]°')
    def residual(q):
        t = fk(q)
        return np.r_[(t[:3,3]-target[:3,3])*10,
                     Rotation.from_matrix(target[:3,:3].T@t[:3,:3]).as_rotvec()]
    bounds=np.asarray(limits)
    fit=least_squares(residual,start,bounds=(bounds[:,0],bounds[:,1]),max_nfev=400)
    d,a=pose_error(fk(fit.x),target)
    if not fit.success:
        raise ValueError(f'IK 拒绝：收敛失败：{fit.message}；无运动')
    if d>.001 or a>np.deg2rad(.2):
        raise ValueError(f'IK 拒绝：残差超限：位置{d*1000:.6f} mm，姿态{np.rad2deg(a):.6f}°；无运动')
    delta = np.rad2deg(fit.x-start)
    if max(abs(delta)) > max_delta_deg:
        axis = int(np.argmax(abs(delta)))
        raise ValueError(f'IK 拒绝：J{axis+1}相对种子变化{delta[axis]:+.3f}°，超过{max_delta_deg:g}°；无运动')
    return fit.x


def diagnose(joints, flange, camera_point, handeye, limits, fk, clearance=.125,
             max_delta_deg=MAX_HOVER_JOINT_DELTA_DEG, max_travel_mm=150.,
             target_offset_m=(0., 0., 0.), tcp_offset_m=None, *, require_fixed_start=True,
             orientation_override=None):
    if not np.isfinite(max_travel_mm) or not 0 < max_travel_mm <= 200:
        raise ValueError('法兰位移上限必须在 (0,200] mm')
    target_offset_m = np.asarray(target_offset_m, float)
    tcp_offset_m = np.asarray(TCP_OFFSET_M if tcp_offset_m is None else tcp_offset_m, float)
    joints=np.asarray(joints,float)
    fixed=fk(np.deg2rad(TARGET_DEG))
    orientation = fixed[:3,:3] if orientation_override is None else np.asarray(orientation_override, float)
    target,base=hover_target(flange,camera_point,handeye,orientation,clearance,
                             target_offset_m,tcp_offset_m)
    tcp=np.eye(4);tcp[:3,3]=tcp_offset_m
    current_pose=matrix_pose(flange); target_pose=matrix_pose(target)
    d,a=pose_error(fk(joints),flange)
    issues=[]
    if d>.002 or a>np.deg2rad(.2): issues.append('当前 CAN 与 piper FK 不一致')
    if require_fixed_start and max(abs(joints-np.deg2rad(TARGET_DEG)))>np.deg2rad(1):
        issues.append('当前不是固定抓取姿态；仅诊断，不允许执行')
    if np.linalg.norm(target[:3,3]-flange[:3,3]) > max_travel_mm/1000:
        issues.append(f'目标法兰位移超过{max_travel_mm:g} mm')
    out=dict(mode='DRY_RUN_ONLY',motion_enabled=False,
        max_hover_joint_delta_deg=max_delta_deg, max_flange_travel_mm=max_travel_mm,
        current_joints_deg=np.rad2deg(joints).tolist(),current_joints_rad=joints.tolist(),
        current_CAN_flange_m_rad=current_pose,
        current_CAN_flange_mm_deg=(np.r_[np.array(current_pose[:3])*1000,np.rad2deg(current_pose[3:])]).tolist(),
        red_block_camera_xyz_m=list(camera_point),red_block_base_xyz_m=base.tolist(),
        T_base_flange=np.asarray(flange).tolist(),T_flange_camera=np.asarray(handeye).tolist(),
        provisional_T_flange_tcp=tcp.tolist(),
        target_offset_mm=(target_offset_m*1000).tolist(),
        tcp_offset_mm=(tcp_offset_m*1000).tolist(),
        fixed_reference_joints_deg=list(TARGET_DEG),fixed_reference_flange_m_rad=matrix_pose(fixed),
        orientation_source=('piper FK at fixed TARGET_DEG, held constant; not from vision'
                            if orientation_override is None else
                            'fixed tool-down axis with optimized rotation about tool Z; not from vision'),
        hypothetical_target_flange_m_rad=target_pose,
        hypothetical_target_flange_mm_deg=np.r_[np.array(target_pose[:3])*1000,np.rad2deg(target_pose[3:])].tolist(),
        hypothetical_target_TCP_m_rad=matrix_pose(target@tcp),clearance_above_detected_surface_mm=clearance*1000,
        CAN_vs_piper_mm_deg=[d*1000,float(np.rad2deg(a))],issues=issues,
        coordinate_convention='T_A_B maps B to A; m/rad internally; R=Rz(yaw)Ry(pitch)Rx(roll)',
        assumptions=['base +Z is upward',f'flange->TCP offset {np.round(tcp_offset_m*1000,1).tolist()} mm in the flange frame; axial 80 mm measured, lateral not calibrated',
                     f'base-frame target trim {np.round(target_offset_m*1000,1).tolist()} mm; empirical, not a calibration',
                     'YOLO center depth is a visible surface point, not necessarily top or grasp center'])
    try:
        q=solve_target(target,joints,limits,fk,max_delta_deg)
        out.update(IK_joints_deg=np.rad2deg(q).tolist(),IK_joints_rad=q.tolist(),
                   IK_delta_from_current_deg=np.rad2deg(q-joints).tolist(),
                   IK_delta_from_fixed_deg=np.rad2deg(q-np.deg2rad(TARGET_DEG)).tolist())
        if max(abs(q-np.deg2rad(TARGET_DEG)))>np.deg2rad(max_delta_deg):
            issues.append(f'IK 相对固定姿态累计变化超过{max_delta_deg:g}°')
    except ValueError as error:
        out.update(IK_joints_deg=None,IK_error=str(error))
        issues.append(str(error))
    return out


def diagnose_best_tool_yaw(*args, **kwargs):
    """Keep the fixed flange attitude when possible; relax tool-Z yaw by at most 45°.

    The ordering is deliberate: attitude deviation is more important than
    reducing an already-acceptable joint excursion.  A solution outside this
    small envelope is rejected instead of turning the wrist around the target.
    """
    fk = args[5]
    fixed_rotation = fk(np.deg2rad(TARGET_DEG))[:3, :3]
    max_delta_deg = float(args[7] if len(args) > 7 else
                          kwargs.get('max_delta_deg', MAX_HOVER_JOINT_DELTA_DEG))

    fixed_report = None
    # Test the unchanged flange attitude first, then the smallest symmetric
    # relaxations.  Never search beyond 45 degrees merely to obtain an IK.
    yaw_order = [0] + [sign*angle for angle in range(5, 46, 5) for sign in (1, -1)]
    for yaw_deg in yaw_order:
        if yaw_deg == 0:
            report = diagnose(*args, **kwargs)
            fixed_report = report
        else:
            rotation = fixed_rotation @ Rotation.from_euler(
                'z', yaw_deg, degrees=True).as_matrix()
            report = diagnose(*args, **kwargs, orientation_override=rotation)
        if report.get('IK_joints_rad') is None:
            continue
        current_delta = np.max(np.abs(report['IK_delta_from_current_deg']))
        fixed_delta = np.max(np.abs(report['IK_delta_from_fixed_deg']))
        if current_delta <= max_delta_deg and fixed_delta <= max_delta_deg:
            report['tool_yaw_from_fixed_deg'] = float(yaw_deg)
            report['tool_yaw_search_deg'] = [-45., 45.]
            report['tool_yaw_selection'] = 'fixed attitude first; then ±5° steps'
            return report

    report = fixed_report if fixed_report is not None else diagnose(*args, **kwargs)
    report['tool_yaw_from_fixed_deg'] = None
    report['tool_yaw_search_deg'] = [-45., 45.]
    report['tool_yaw_selection'] = 'fixed attitude first; then ±5° steps'
    report['issues'].append(
        f'固定法兰姿态及工具轴±45°范围内无满足单关节变化≤{max_delta_deg:g}°的IK解')
    return report


def depth_diagnostic(frame, box, pixel_uv):
    depth=np.asarray(frame.depth_m)
    h,w=depth.shape
    u,v=pixel_uv
    def stats(values):
        finite=np.isfinite(values)
        positive=finite & (values>0)
        accepted=finite & (values>=.07) & (values<=.6)
        valid=values[positive]
        return dict(pixel_count=int(values.size),positive_fraction=float(positive.mean()),
            accepted_70_600mm_fraction=float(accepted.mean()),
            positive_min_mm=None if not valid.size else float(valid.min()*1000),
            positive_median_mm=None if not valid.size else float(np.median(valid)*1000),
            positive_max_mm=None if not valid.size else float(valid.max()*1000))
    raw=float(depth[v,u])
    reason=('nonfinite' if not np.isfinite(raw) else 'zero_or_negative' if raw<=0
            else 'below_70mm' if raw<.07 else 'above_600mm' if raw>.6 else 'in_range')
    x1,y1,x2,y2=np.asarray(box,dtype=int)
    roi=depth[max(0,y1):min(h,y2+1),max(0,x1):min(w,x2+1)]
    return dict(center_uv=[int(u),int(v)],center_raw_m=raw if np.isfinite(raw) else None,
        center_raw_mm=raw*1000 if np.isfinite(raw) else None,center_category=reason,
        neighborhood_11x11=stats(depth[max(0,v-5):min(h,v+6),max(0,u-5):min(w,u+6)]),
        box_depth=stats(roi),whole_frame=stats(depth),
        depth_scale_m=frame.metadata.get('depth_scale_m'),
        note='Diagnostic only: no hole filling, fallback point selection or threshold change')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channel',default='can1')
    parser.add_argument('--execute',action='store_true',help='已禁用：传入即拒绝')
    parser.add_argument('--dry-run',action='store_true',help='只读模式（默认且唯一模式）')
    parser.add_argument('--clearance-mm',type=float,default=125)
    parser.add_argument('--input',type=Path,help='离线 JSON：joint_rad,T_base_flange,xyz_camera_m')
    parser.add_argument('--output',type=Path,default=Path('yolo_grasp/runs')/time.strftime('coordinate_audit_%Y%m%d_%H%M%S'))
    args=parser.parse_args(argv)
    if args.execute: parser.error('实际运动已禁用；只能 --dry-run，不会使能、运动或复位')
    if not np.isfinite(args.clearance_mm) or not 10<=args.clearance_mm<=150:
        parser.error('--clearance-mm 必须在10–150之间（原审计包络为100–150）')
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh,get_mdh
    root=Path(__file__).resolve().parents[2]
    raw=(root/'data/handeye/d405_02_fixed_split/candidate_result.json').read_bytes()
    calibration=json.loads(raw)
    import os
    if calibration['camera_serial']!=os.environ.get('GRASP_CAMERA_SERIAL','260322275595'):
        raise ValueError('手眼相机身份不匹配（如需换相机，设置环境变量 GRASP_CAMERA_SERIAL）')
    x=transform(calibration['T_flange_wrist'])
    mdh=get_mdh('piper');fk=lambda q:pose_matrix(fk_from_mdh(mdh,list(q)))
    limits=piper_planning_limits(ROBOT_JOINT_LIMIT_PRESET_RAD)
    args.output.mkdir(parents=True,exist_ok=False)
    print('DRY RUN：不发送任何 CAN 控制帧；不会自动到固定点。')
    if args.input:
        sample=json.loads(args.input.read_text())
    else:
        from .handeye_capture import PoseReceiver,PoseHistory
        from .camera import RealSense
        from .live_depth import locate
        from ultralytics import YOLO
        import cv2
        receiver=PoseReceiver(args.channel);history=PoseHistory(receiver)
        try:
            model=YOLO(str(root/'models/red_block_best.pt'))
            stream=dict(width=640,height=480,fps=30)
            cfg=dict(serial=os.environ.get('GRASP_CAMERA_SERIAL','260322275595'),
                     color=stream,depth=stream,warmup_frames=30,timeout_ms=1000)
            with RealSense(cfg) as camera:
                print('相机窗口 S=打印/保存只读诊断；Q/Esc=退出（不控制机械臂）')
                while True:
                    start=time.time();frame=camera.capture(discard=2)
                    prediction=model.predict(source=frame.bgr,device='0',imgsz=640,conf=.6,verbose=False)[0]
                    cv2.imshow('D405 DRY RUN - S audit, Q exit',prediction.plot())
                    key=cv2.waitKey(1)&0xff
                    if key in (ord('q'),27): return 0
                    if key!=ord('s'):continue
                    try:
                        boxes=[b for b in prediction.boxes if model.names[int(b.cls.item())]=='red_block']
                        if len(boxes)!=1:raise ValueError('必须且只能检测到一个 red_block')
                        dep=locate(frame,boxes[0].xyxy[0].cpu().numpy(),min_depth=.07,max_depth=.6)
                        if not dep['valid']:
                            if dep['pixel_uv'] is not None and dep['reason']!='center_out_of_bounds':
                                report=depth_diagnostic(frame,boxes[0].xyxy[0].cpu().numpy(),dep['pixel_uv'])
                                report['rejection']=dep
                                directory=args.output / ('depth_rejected_'+str(time.time_ns()))
                                frame.save(directory)
                                (directory/'depth_diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
                                cv2.imwrite(str(directory/'detection.png'),prediction.plot())
                                depth=frame.depth_m
                                colors=cv2.applyColorMap((np.clip(np.nan_to_num(depth,nan=0,posinf=0,neginf=0)/.6,0,1)*255).astype(np.uint8),cv2.COLORMAP_TURBO)
                                colors[~np.isfinite(depth)|(depth<=0)]=0
                                cv2.drawMarker(colors,tuple(dep['pixel_uv']),(255,255,255),cv2.MARKER_CROSS,16,1)
                                cv2.imwrite(str(directory/'depth_0_600mm.png'),colors)
                                cv2.imshow('Depth diagnostic 0-600mm; black=invalid',colors)
                                print(json.dumps(report,indent=2,ensure_ascii=False))
                                print(f'深度诊断已保存：{directory}')
                            raise ValueError(str(dep))
                        time.sleep(.25);robot,bracket=history.window(start-.3,time.time())
                        sample=dict(joint_rad=robot['joints_rad'],T_base_flange=robot['T_base_flange'],
                                    xyz_camera_m=dep['xyz_camera_m'],robot_bracket=bracket)
                        frame.save(args.output/'observation')
                        cv2.imwrite(str(args.output/'detection.png'),prediction.plot())
                        break
                    except ValueError as error:print(f'拒绝观测：{error}')
        finally:
            history.close();receiver.close();cv2.destroyAllWindows()
    result=diagnose(sample['joint_rad'],transform(sample['T_base_flange']),sample['xyz_camera_m'],x,limits,fk,args.clearance_mm/1000)
    result['handeye_sha256']=hashlib.sha256(raw).hexdigest()
    result['handeye_validated']=calibration.get('validated',False)
    (args.output/'input.json').write_text(json.dumps(sample,indent=2)+'\n')
    (args.output/'diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))
    print(f'仅计算，无目标已发送。诊断保存：{args.output}')
    return 0
