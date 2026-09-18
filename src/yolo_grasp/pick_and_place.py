"""One session: fixed pose -> grasp the red block -> lift -> fixed pose -> hover the black frame -> release -> home.

This entry adds no new motion logic. It runs the two audited legs in order:

1. ``hover_red_block.py --grasp``: fixed grasp pose (3%), red-block observation,
   typed ``go``, straight approach, slow close on the block, lift along base +Z.
   It ends in the enabled hold; type ``quit`` there to keep the block and continue
   (``home`` would return to the initial pose and end this flow).
2. ``place_on_frame.py --open-gripper``: fixed grasp pose again (3%), black-frame
   observation, typed ``go``, level move above the frame, attitude turn, vertical
   drop, then release the block and retreat along base +Z. The remaining hold
   answers ``home`` (fixed pose, then the saved initial pose).

Both legs keep their own guards, ``go`` confirmations and stop keys; the gripper is
only commanded by the grasp leg's close and by the placement leg's ``--open-gripper``.
"""
import argparse
import gc
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from yolo_grasp import hover_red_block, place_on_frame
from yolo_grasp.hover_red_block import available_memory_mb


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--channel', default='can1')
    parser.add_argument('--imgsz', type=int, choices=(320, 416, 512, 640), default=320,
                        help='默认320（现场用值，省内存）；想更准可给 640')
    parser.add_argument('--device', default='cpu',
                        help='默认cpu（抓取腿 YOLO 不需要 CUDA，避免 CUBLAS_STATUS_ALLOC_FAILED）；GPU 给 0')
    parser.add_argument('--no-window', action='store_true', help='两条腿都不打开相机窗口')
    parser.add_argument('--min-available-mb', type=float, default=1200,
                        help='启动内存下限(MB)，两条腿各自检查；默认1200（现场用值）')
    parser.add_argument('--hold-tolerance-deg', type=float, default=None,
                        help='停在固定点等待时的允许偏离(°)，默认沿用两条腿各自的值(0.8)')
    parser.add_argument('--grasp-clearance-mm', type=float, default=15,
                        help='抓取腿：TCP 停在红块表面上方高度(mm)，默认15，到位后夹爪才慢慢闭合')
    parser.add_argument('--tracking-lock-height-mm', type=float, default=45,
                        help='抓取腿：TCP下降到红块顶面上方此高度后锁定轨迹，默认45')
    parser.add_argument('--grip-force-n', type=float, default=2.,
                        help='夹住物块的力参数(N)，默认2；必须是现场验证过的值')
    parser.add_argument('--lift-to-mm', type=float, default=100,
                        help='抬升后停在物块上方多少毫米，默认100')
    parser.add_argument('--grasp-max-joint-delta-deg', type=float, default=90,
                        help='抓取腿：单关节变化上限(°)，默认90（现场用值；物块靠左/靠后时 J6 要摆动 50–65°）')
    parser.add_argument('--grasp-max-travel-mm', type=float, default=200,
                        help='抓取腿：法兰总位移上限(mm)，默认200（现场用值）')
    parser.add_argument('--target-move-mm', type=float, default=50.,
                        help='抓取接近中红块大幅移动检测阈值(mm)，默认50')
    parser.add_argument('--max-replans', type=int, default=5,
                        help='抓取接近最多自动重规划次数，默认5')
    parser.add_argument('--grasp-target-offset-mm', type=float, nargs=3,
                        default=[-43., -28., 0.],
                        metavar=('DX', 'DY', 'DZ'),
                        help='抓取腿：基座系目标微调(mm)，默认 -43 -28 0（2026-09-18 现场确认）；'
                             '只传给抓取腿，放置腿用自己的目标')
    parser.add_argument('--place-clearance-mm', type=float, default=55,
                        help='放置腿：TCP 停在黑框平面上方高度(mm)，默认55（2026-09-18 由50增加5 mm）')
    parser.add_argument('--max-j5-deg', type=float, default=60,
                        help='放置腿：J5 规划上限(°)，默认60（J5 机械临界约67°）')
    parser.add_argument('--max-tilt-deg', type=float, default=25,
                        help='放置腿：允许的最大工具轴倾斜(°)，默认25')
    parser.add_argument('--open-mm', type=float, default=None,
                        help='松爪目标开度(mm)，默认不指定=设备最大行程')
    parser.add_argument('--place-target-offset-mm', type=float, nargs=3, default=None,
                        metavar=('DX', 'DY', 'DZ'),
                        help='放置腿：基座系目标微调(mm)，默认沿用放置腿的 5 5 0（沿基座 X、Y 正方向各补偿5 mm）')
    parser.add_argument('--open-force-n', type=float, default=2.,
                        help='松爪力参数(N)，默认2')
    parser.add_argument('--retract-mm', type=float, default=30,
                        help='松爪后沿基座 +Z 抬离高度(mm)，默认30')
    return parser


def grasp_stage_argv(args):
    """argv for ``hover_red_block.main``: fixed pose -> grasp -> lift."""
    argv = ['--channel', args.channel, '--grasp', '--continue-to-place',
            '--clearance-mm', str(args.grasp_clearance_mm),
            '--tracking-lock-height-mm', str(args.tracking_lock_height_mm),
            '--target-offset-mm', *[str(v) for v in args.grasp_target_offset_mm],
            '--grip-force-n', str(args.grip_force_n),
            '--lift-to-mm', str(args.lift_to_mm),
            '--max-joint-delta-deg', str(args.grasp_max_joint_delta_deg),
            '--max-travel-mm', str(args.grasp_max_travel_mm),
            '--target-move-mm', str(args.target_move_mm),
            '--max-replans', str(args.max_replans),
            '--imgsz', str(args.imgsz), '--device', args.device,
            '--min-available-mb', str(args.min_available_mb)]
    if args.hold_tolerance_deg is not None:
        argv += ['--hold-tolerance-deg', str(args.hold_tolerance_deg)]
    if args.no_window:
        argv.append('--no-window')
    return argv


# Probing window between the two legs. 30 probes with a 0.5 s pause (and a
# gc.collect() each round) meant 22 silent seconds on 2026-09-15 (13:35, 15:57,
# 16:42, 16:56), and it decided nothing: on timeout the placement leg is started
# with the gate disabled anyway, exactly as documented below. Waiting longer
# cannot help either -- gc.collect() drops Python objects, but the interpreter's
# arenas and the cached heap are usually not handed back to the OS, so
# MemAvailable does not climb. Six probes of 0.25 s bound the wait to ~1.5 s and
# every probe is printed.
MEMORY_PROBES = 6
MEMORY_PROBE_PAUSE_S = .25


def wait_for_memory(min_mb, tries=MEMORY_PROBES, pause=MEMORY_PROBE_PAUSE_S,
                    sleep=time.sleep, report=None):
    """Let the grasp stage's torch/camera allocations go before the next check.

    Both legs run in this one process, so right after the grasp phase the memory
    the YOLO model and the camera held may still be unreleased; the placement leg
    would then refuse to start (2026-09-14: 1041 MB available vs a 1200 MB gate).
    Returns the last measured available memory in MB.

    ``report(attempt, free_mb)`` lets the caller show every probe, so a low
    reading is never a silent stall.
    """
    free = None
    for attempt in range(1, max(1, tries) + 1):
        gc.collect()
        free = available_memory_mb()['MemAvailable']
        if report is not None:
            report(attempt, free)
        if free >= min_mb:
            return free
        if attempt < tries:
            sleep(pause)
    return free


def place_stage_argv(args, min_available_mb=None):
    """argv for ``place_on_frame.main``: fixed pose -> hover -> release."""
    gate = args.min_available_mb if min_available_mb is None else min_available_mb
    argv = ['--channel', args.channel, '--open-gripper',
            '--clearance-mm', str(args.place_clearance_mm),
            '--max-j5-deg', str(args.max_j5_deg),
            '--max-tilt-deg', str(args.max_tilt_deg),
            '--open-force-n', str(args.open_force_n),
            '--retract-mm', str(args.retract_mm),
            '--imgsz', str(args.imgsz), '--device', args.device,
            '--min-available-mb', str(gate)]
    if args.open_mm is not None:
        argv += ['--open-mm', str(args.open_mm)]
    if args.place_target_offset_mm is not None:
        argv += ['--target-offset-mm', *[str(v) for v in args.place_target_offset_mm]]
    if args.hold_tolerance_deg is not None:
        argv += ['--hold-tolerance-deg', str(args.hold_tolerance_deg)]
    if args.no_window:
        argv.append('--no-window')
    return argv


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not sys.stdin.isatty():
        raise SystemExit('需要交互终端：两条腿都要输入 go，并接收停止键')
    print('=' * 72, flush=True)
    print('完整抓放：固定点 → 识别红块 → go → 接近 → 夹紧 → 抬升 → 固定点 →', flush=True)
    print('          识别黑框 → go → 到框上方 → 松爪 → 抬离 → home 回初始', flush=True)
    print('阶段 1 抬起后停在"保持"：输入 place = 带着物块继续放置'
          '（先回固定抓取点、再识别黑框）；输入 home = 回初始位置并结束本次流程；'
          'quit = 就地保持退出。', flush=True)
    if args.device == '0':
        print('提示：抓取腿用 YOLO，需要 CUDA 上下文；内存紧张时报 '
              'CUDA error: CUBLAS_STATUS_ALLOC_FAILED，此时改用 --device cpu --imgsz 320。', flush=True)
    print('=' * 72, flush=True)
    status = hover_red_block.main(grasp_stage_argv(args))
    if status == hover_red_block.PLACE_CONTINUE:
        print('阶段 1 收到 place：带着物块继续放置。', flush=True)
    elif status:
        print(f'抓取阶段未正常结束（返回 {status}），不进入放置阶段。', flush=True)
        return status
    else:
        print('在抓取保持里结束了流程（quit / home），不进入放置阶段。', flush=True)
        return 0
    print('=' * 72, flush=True)
    print('阶段 2：放置。机械臂会先回固定抓取点，再识别黑框、等你输入 go。', flush=True)
    gate = args.min_available_mb
    wait_start = time.monotonic()
    free = wait_for_memory(gate or 0, report=lambda attempt, mb: print(
        f'  内存探测 {attempt}/{MEMORY_PROBES}：可用 {mb:.0f} MB（门槛 {gate:g} MB，'
        f'探测间隔 {MEMORY_PROBE_PAUSE_S:g} 秒）', flush=True))
    if gate and free < gate:
        print(f'注意：抓取阶段结束后可用内存 {free:.0f} MB，低于 {gate:g} MB 的门槛'
              '（抓取腿的 torch/相机内存可能还没释放，桌面程序也占着不少）。'
              f'探测 {MEMORY_PROBES} 次后不再等（继续等 MemAvailable 也不会回升），'
              '本次放置阶段改用 --min-available-mb 0 继续；相机/预览若因内存不足会自行报错。',
              flush=True)
    print(f'放置阶段可用内存 {free:.0f} MB（等待 {time.monotonic()-wait_start:.1f} 秒）。', flush=True)
    print('=' * 72, flush=True)
    return place_on_frame.main(place_stage_argv(args, min_available_mb=0 if free < gate else gate))


if __name__ == '__main__':
    raise SystemExit(main())
