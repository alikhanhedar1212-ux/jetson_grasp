"""Supervised small joint steps to a freshly audited hover. No grasp commands."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from grasp import arm_console as ac
from grasp.j6_experiment import Controller,Receiver,query_firmware,snapshot
from grasp.geometry import pose_matrix,pose_error
from grasp.hover_motion import (GUARD_HARD_MARGIN_DEG,GUARD_NOISE_FLOOR_DEG,
                               GUARD_PERSIST_SAMPLES,HOLD_TOLERANCE_DEG,
                               MOTION_TOLERANCE_DEG,ZERO_ZONE_DEG,ZERO_ZONE_MAX_ERROR_DEG)


class GuardedController(Controller):
    guard_start=None
    guard_goal=None
    # Tight while a commanded move is in progress; a parked hold may widen it
    # through the attribute so the documented 0.3-0.4 deg snap does not stop
    # the session.
    guard_tolerance_deg=MOTION_TOLERANCE_DEG
    # Consecutive out-of-band samples needed to stop, and the optional callback
    # used to record ignored single-sample excursions.
    guard_strikes=0
    guard_strike_hook=None
    def tick(self):
        if self.guard_goal is not None and not self.locked:
            try:
                state=self.state();state.healthy()
                self.log(dict(event='feedback',joint_rad=state.joints.tolist(),stamps=state.stamps.tolist()))
                tolerance=np.deg2rad(self.guard_tolerance_deg)
                lo=np.minimum(self.guard_start,self.guard_goal)-tolerance
                hi=np.maximum(self.guard_start,self.guard_goal)+tolerance
                raw=np.maximum(lo-state.joints,state.joints-hi)
                if np.max(raw)>np.deg2rad(GUARD_HARD_MARGIN_DEG):
                    raise RuntimeError(f'实际关节偏离本步起点—目标范围超过{self.guard_tolerance_deg:g}°'
                                       f'+{GUARD_HARD_MARGIN_DEG:g}°硬界限（单帧 {np.rad2deg(np.max(raw)):.3f}°）')
                # The joint frames report exactly 0.000 inside the documented
                # zero zone while the motor frames show up to ~0.4 deg there
                # (verified 2026-09-11 11:30), so that reading is not evidence
                # of movement. Counts of a few 0.001 deg are noise as well.
                untrusted=((np.abs(state.joints)<=np.deg2rad(ZERO_ZONE_DEG))
                           &(raw<=np.deg2rad(ZERO_ZONE_MAX_ERROR_DEG)))
                excursion=float(np.max(np.where(untrusted,0.,raw)))
                if excursion>np.deg2rad(GUARD_NOISE_FLOOR_DEG):
                    self.guard_strikes+=1
                    if self.guard_strikes>=GUARD_PERSIST_SAMPLES:
                        raise RuntimeError(f'连续{self.guard_strikes}帧实际关节偏离本步起点—目标范围'
                                           f'超过{self.guard_tolerance_deg:g}°')
                    if self.guard_strike_hook is not None:
                        self.guard_strike_hook(dict(strike=self.guard_strikes,
                            excursion_deg=float(np.rad2deg(excursion)),
                            tolerance_deg=self.guard_tolerance_deg,
                            joint_deg=np.rad2deg(state.joints).tolist()))
                else:
                    self.guard_strikes=0
                if not all(state.enabled) or state.ctrl_mode!=1 or state.mode_feedback!=1:
                    raise RuntimeError('使能或 CAN/J 模式丢失')
            except BaseException as error:
                self.stop(str(error));raise
        super().tick()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--diagnostic',type=Path,required=True)
    p.add_argument('--channel',default='can1');p.add_argument('--execute',action='store_true')
    p.add_argument('--max-travel-mm',type=float,default=150,help='法兰总位移上限，默认150 mm；现场确认后最多可设200 mm')
    args=p.parse_args()
    if not np.isfinite(args.max_travel_mm) or not 0<args.max_travel_mm<=200:raise ValueError('法兰位移上限必须在 (0,200] mm')
    if args.max_travel_mm>150:print(f'法兰位移上限已显式扩大到 {args.max_travel_mm:g} mm，请确认整段路径空间。')
    r=json.loads(args.diagnostic.read_text())
    if r.get('mode')!='DRY_RUN_ONLY' or not 100<=r['clearance_above_detected_surface_mm']<=150:raise ValueError('需要100–150mm上方目标的dry-run记录')
    from pyAgxArm.utiles.mdh_kinematics import fk_from_mdh,get_mdh
    from pyAgxArm.api.constants import ROBOT_JOINT_LIMIT_PRESET_RAD
    from grasp.hover_test import solve_target
    from grasp.mat_pose import TARGET_DEG
    q0=np.array(r['current_joints_rad']);start=pose_matrix(r['current_CAN_flange_m_rad']);target=pose_matrix(r['hypothetical_target_flange_m_rad'])
    mdh=get_mdh('piper');fk=lambda q:pose_matrix(fk_from_mdh(mdh,list(q)))
    limits=list(ROBOT_JOINT_LIMIT_PRESET_RAD['piper_x'].values())
    if max(abs(q0-np.deg2rad(TARGET_DEG)))>np.deg2rad(1):raise ValueError('记录不在固定抓取姿态')
    if pose_error(target,fk(np.deg2rad(TARGET_DEG)))[1]>1e-6:raise ValueError('目标姿态不是固定姿态')
    if np.linalg.norm(target[:3,3]-start[:3,3])>args.max_travel_mm/1000:raise ValueError(f'总位移超过{args.max_travel_mm:g}mm')
    path=[];q=q0.copy()
    for f in np.linspace(0,1,151):
        t=target.copy();t[:3,3]=(1-f)*start[:3,3]+f*target[:3,3]
        next_q=solve_target(t,q,limits,fk)
        if max(abs(next_q-q))>np.deg2rad(.5):raise ValueError('单步关节变化超过0.5°')
        if max(abs(next_q-q0))>np.deg2rad(35):raise ValueError('累计关节变化超过35°')
        path.append(next_q);q=next_q
    print(f'计划 {len(path)} 个小步；终点角度°={np.rad2deg(path[-1]).round(3)}；只到上方，不下降闭爪。')
    if not args.execute:print('仅离线计划；未连接CAN。');return
    if time.time()-args.diagnostic.stat().st_mtime>300:raise ValueError('诊断超过5分钟，请重新采集；物块不可移动')
    if not sys.stdin.isatty():raise ValueError('需要交互终端')
    from pyAgxArm import AgxArmFactory,create_agx_arm_config,resolve_firmware_profile
    output=args.diagnostic.parent/('steps_'+time.strftime('%H%M%S'));output.mkdir(exist_ok=False)
    arm=receiver=c=None
    with (output/'events.jsonl').open('x',buffering=1) as log:
        def event(row):log.write(json.dumps(dict(host_s=time.time(),**row))+'\n');log.flush()
        try:
            with ac.shutdown_signals():
                def create(profile):return AgxArmFactory.create_arm(create_agx_arm_config(robot='piper_x',firmeware_version=profile,interface='socketcan',channel=args.channel))
                arm=create('default');arm.connect();firmware=query_firmware(arm);arm.disconnect()
                arm=create(resolve_firmware_profile('piper_x',firmware['software_version']));receiver=Receiver(args.channel);arm.connect();time.sleep(.5)
                c=GuardedController(arm,receiver,'/tmp/unused_home.json',{},limits,speed=3,timeout=10);c.log=event
                state,flange,stamps=snapshot(receiver,time.time())
                if not all(state.enabled) or state.ctrl_mode!=1 or state.mode_feedback!=1:raise ValueError('须已在固定点停稳并保持使能/CAN-J模式；本程序不自动使能或切模式')
                if max(abs(state.joints-q0))>np.deg2rad(.2):raise ValueError('当前关节与新诊断不一致')
                d,a=pose_error(flange,start)
                if d>.002 or a>np.deg2rad(.2):raise ValueError('当前法兰与新诊断不一致')
                c.prepared=True;c.guard_start=state.joints.copy();c.guard_goal=state.joints.copy()
                # Log actual SDK transmission boundary without changing encoding.
                # Prepare cached speed without an early standalone mode frame.
                arm._msg_mode.move_spd_rate_ctrl=3
                original_move=arm.move_j
                def logged_move(joints):
                    event(dict(event='SDK_move_j_call',joint_rad=list(joints),joint_deg=np.rad2deg(joints).tolist()))
                    original_move(joints)
                    event(dict(event='SDK_move_j_return'))
                arm.move_j=logged_move
                event(dict(event='plan',diagnostic=r,joint_path_rad=[q.tolist() for q in path]))
                import select,os
                with ac.keyboard() as fd:
                    for i,q in enumerate(path):
                        print(f'步 {i+1}/{len(path)}：目标°={np.rad2deg(q).round(3)}；输入 next 回车执行；Esc/空格停止。',flush=True)
                        pending=''
                        while True:
                            c.tick()
                            if c.locked:raise RuntimeError('会话锁定')
                            if not select.select([fd],[],[],.02)[0]:continue
                            key=os.read(fd,1)
                            if key in (b'\x1b',b' ',b'',b'\x03'):raise KeyboardInterrupt()
                            if key in (b'\r',b'\n'):
                                if pending=='next':break
                                pending=''
                            else:pending+=key.decode(errors='ignore')
                        c.guard_start=c.state().joints.copy();c.guard_goal=q.copy()
                        if max(abs(q-c.guard_start))>np.deg2rad(.8):raise ValueError('当前到下一步增量超过0.8°')
                        c.experiment_target=q.copy();event(dict(event='target_requested',joint_rad=q.tolist(),flange_m_rad=r['hypothetical_target_flange_m_rad']))
                        c.command('home')
                        while c.active:
                            if select.select([fd],[],[],0)[0]:
                                key=os.read(fd,1)
                                if key in (b'\x1b',b' ',b'',b'\x03'):raise KeyboardInterrupt()
                            c.tick()
                            if c.locked:raise RuntimeError('停止')
                            time.sleep(.02)
                        event(dict(event='step_completed',index=i,joint_rad=c.state().joints.tolist()))
                print('已完成上方试验，保持使能；程序退出后不再监控。')
        except BaseException as error:
            event(dict(event='aborted',error=str(error)))
            if c is not None and not c.locked:c.stop(str(error))
            raise
        finally:
            if receiver is not None:receiver.close()
            if arm is not None:arm.disconnect()
if __name__=='__main__':main()
