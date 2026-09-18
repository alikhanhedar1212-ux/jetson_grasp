import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pick_and_place as m


def test_combined_defaults_match_the_two_audited_legs():
    args = m.build_parser().parse_args([])
    # 抓取阶段用现场验证过的参数（2026-09-14）：
    # --clearance-mm 15 --max-joint-delta-deg 90 --max-travel-mm 200 --target-offset-mm -43 -28 0
    assert (args.grasp_clearance_mm, args.grip_force_n, args.lift_to_mm) == (15., 2., 100.)
    assert (args.grasp_max_joint_delta_deg, args.grasp_max_travel_mm) == (90., 200.)
    assert list(args.grasp_target_offset_mm) == [-43., -28., 0.]
    assert (args.place_clearance_mm, args.max_j5_deg, args.max_tilt_deg) == (55., 60., 25.)
    assert (args.open_mm, args.open_force_n, args.retract_mm) == (None, 2., 30.)
    assert (args.device, args.imgsz, args.min_available_mb) == ('cpu', 320, 1200.)


def test_grasp_stage_is_the_grasp_leg_with_grasp_and_lift():
    args = m.build_parser().parse_args(['--channel', 'can0', '--grasp-clearance-mm', '15',
                                        '--grip-force-n', '3', '--lift-to-mm', '80',
                                        '--grasp-max-joint-delta-deg', '90',
                                        '--grasp-max-travel-mm', '200',
                                        '--min-available-mb', '1200', '--no-window'])
    argv = m.grasp_stage_argv(args)
    assert argv[:3] == ['--channel', 'can0', '--grasp']
    assert float(argv[argv.index('--lift-to-mm') + 1]) == 80.
    assert float(argv[argv.index('--clearance-mm') + 1]) == 15.
    assert float(argv[argv.index('--grip-force-n') + 1]) == 3.
    assert float(argv[argv.index('--max-joint-delta-deg') + 1]) == 90.
    assert float(argv[argv.index('--max-travel-mm') + 1]) == 200.
    assert float(argv[argv.index('--min-available-mb') + 1]) == 1200.
    assert '--no-window' in argv and '--open-gripper' not in argv
    assert '--continue-to-place' in argv


def test_tracking_controls_forwarded_only_to_grasp():
    args = m.build_parser().parse_args(['--target-move-mm', '12', '--max-replans', '3'])
    argv = m.grasp_stage_argv(args)
    assert argv[argv.index('--target-move-mm')+1] == '12.0'
    assert argv[argv.index('--max-replans')+1] == '3'
    assert '--target-move-mm' not in m.place_stage_argv(args)


def test_combined_entry_forwards_the_field_parameters_to_both_legs():
    """The whole run must use the 2026-09-18 grasp values and the frame defaults."""
    args = m.build_parser().parse_args([])
    def val(argv, flag):
        return float(argv[argv.index(flag)+1])
    grasp = m.grasp_stage_argv(args)
    assert val(grasp, '--clearance-mm') == 15.
    assert val(grasp, '--tracking-lock-height-mm') == 45.
    assert [float(v) for v in
            grasp[grasp.index('--target-offset-mm')+1:grasp.index('--target-offset-mm')+4]] == \
        [-43., -28., 0.]
    assert (val(grasp, '--lift-to-mm'), val(grasp, '--grip-force-n')) == (100., 2.)
    assert (val(grasp, '--max-joint-delta-deg'), val(grasp, '--max-travel-mm')) == (90., 200.)
    assert (val(grasp, '--target-move-mm'), val(grasp, '--max-replans')) == (50., 5.)
    assert '--continue-to-place' in grasp and '--open-gripper' not in grasp
    place = m.place_stage_argv(args)
    assert val(place, '--clearance-mm') == 55.
    assert (val(place, '--max-j5-deg'), val(place, '--max-tilt-deg')) == (60., 25.)
    assert (val(place, '--open-force-n'), val(place, '--retract-mm')) == (2., 30.)
    # The placement leg keeps its own 5 5 0 base-frame trim unless asked otherwise.
    assert '--target-offset-mm' not in place
    assert '--open-gripper' in place and '--grasp' not in place


def test_place_stage_releases_and_uses_the_placement_parameters():
    args = m.build_parser().parse_args(['--place-clearance-mm', '45', '--max-j5-deg', '62',
                                        '--open-mm', '60', '--retract-mm', '20'])
    argv = m.place_stage_argv(args)
    assert '--open-gripper' in argv
    assert float(argv[argv.index('--clearance-mm') + 1]) == 45.
    assert float(argv[argv.index('--max-j5-deg') + 1]) == 62.
    assert float(argv[argv.index('--open-mm') + 1]) == 60.
    assert float(argv[argv.index('--retract-mm') + 1]) == 20.
    assert '--grasp' not in argv


def test_open_mm_is_only_forwarded_when_asked(monkeypatch):
    args = m.build_parser().parse_args([])
    assert '--open-mm' not in m.place_stage_argv(args)
    # 抓取腿包络与目标微调按现场值转发，放置腿不受影响
    grasp = m.grasp_stage_argv(args)
    assert float(grasp[grasp.index('--max-joint-delta-deg') + 1]) == 90.
    assert float(grasp[grasp.index('--max-travel-mm') + 1]) == 200.
    assert grasp[grasp.index('--target-offset-mm') + 1:grasp.index('--target-offset-mm') + 4] == \
        ['-43.0', '-28.0', '0.0']
    assert '--target-offset-mm' not in m.place_stage_argv(args)


def test_main_runs_grasp_then_place(monkeypatch):
    calls = []
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(m.hover_red_block, 'main',
                        lambda argv: calls.append(('grasp', list(argv))) or m.hover_red_block.PLACE_CONTINUE)
    monkeypatch.setattr(m.place_on_frame, 'main', lambda argv: calls.append(('place', list(argv))) or 0)
    assert m.main(['--channel', 'can1']) == 0
    assert [name for name, _ in calls] == ['grasp', 'place']
    assert '--grasp' in calls[0][1] and '--continue-to-place' in calls[0][1]
    assert '--open-gripper' in calls[1][1]


def test_main_stops_when_the_operator_ends_the_grasp_hold(monkeypatch):
    """quit / home at the grasp hold (return code 0) must not start the placement."""
    calls = []
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(m.hover_red_block, 'main', lambda argv: calls.append('grasp') or 0)
    monkeypatch.setattr(m.place_on_frame, 'main', lambda argv: calls.append('place') or 0)
    assert m.main([]) == 0
    assert calls == ['grasp']


def test_memory_dip_between_stages_does_not_block_the_placement(monkeypatch):
    """Regression for the 1041 MB / 1200 MB refusal right after the grasp stage."""
    calls = []
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(m.hover_red_block, 'main',
                        lambda argv: calls.append(('grasp', list(argv))) or m.hover_red_block.PLACE_CONTINUE)
    monkeypatch.setattr(m.place_on_frame, 'main', lambda argv: calls.append(('place', list(argv))) or 0)
    monkeypatch.setattr(m, 'available_memory_mb', lambda: {'MemAvailable': 1041.})
    monkeypatch.setattr(m, 'wait_for_memory', lambda min_mb, **k: m.available_memory_mb()['MemAvailable'])
    assert m.main(['--min-available-mb', '1200']) == 0
    place_argv = calls[1][1]
    assert float(place_argv[place_argv.index('--min-available-mb') + 1]) == 0.
    grasp_argv = calls[0][1]
    assert float(grasp_argv[grasp_argv.index('--min-available-mb') + 1]) == 1200.


def test_wait_for_memory_gives_cached_allocations_time(monkeypatch):
    values = iter([900., 1100., 1300.])
    monkeypatch.setattr(m, 'available_memory_mb', lambda: {'MemAvailable': next(values)})
    slept = []
    assert m.wait_for_memory(1200., tries=5, pause=.01, sleep=slept.append) == 1300.
    assert slept == [.01, .01]
    monkeypatch.setattr(m, 'available_memory_mb', lambda: {'MemAvailable': 800.})
    assert m.wait_for_memory(1200., tries=3, pause=.01, sleep=lambda s: None) == 800.


def test_memory_probe_window_is_short_and_visible(monkeypatch):
    """2026-09-15 现场：30 次 gc+0.5 秒的静默探测白等 22 秒（13:35/15:57/16:42/16:56）。

    gc.collect() 只释放 Python 对象，堆一般不还给系统，所以 MemAvailable 不会回升；
    超时后放置腿照样以门槛关闭启动。默认窗口因此缩到 6 次 × 0.25 秒，并逐次打印。
    """
    monkeypatch.setattr(m, 'available_memory_mb', lambda: {'MemAvailable': 1041.})
    probes, slept = [], []
    assert m.wait_for_memory(1200., sleep=slept.append,
                             report=lambda attempt, free: probes.append((attempt, free))) == 1041.
    assert (m.MEMORY_PROBES, m.MEMORY_PROBE_PAUSE_S) == (6, .25)
    assert probes == [(i, 1041.) for i in range(1, 7)]
    assert slept == [.25]*5                      # 最多等 1.5 秒，不是 15 秒


def test_low_memory_prints_every_probe_then_starts_the_placement_leg(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(m.hover_red_block, 'main', lambda argv: m.hover_red_block.PLACE_CONTINUE)
    monkeypatch.setattr(m.place_on_frame, 'main', lambda argv: calls.append(list(argv)) or 0)
    monkeypatch.setattr(m, 'available_memory_mb', lambda: {'MemAvailable': 1041.})
    real = m.wait_for_memory
    monkeypatch.setattr(m, 'wait_for_memory',
                        lambda min_mb, **kw: real(min_mb, sleep=lambda s: None, **kw))
    assert m.main([]) == 0
    out = capsys.readouterr().out
    assert out.count('内存探测') == m.MEMORY_PROBES
    assert '探测 6 次后不再等' in out and '等待 ' in out
    assert float(calls[0][calls[0].index('--min-available-mb') + 1]) == 0.


def test_main_stops_when_the_grasp_stage_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(m.hover_red_block, 'main', lambda argv: calls.append('grasp') or 2)
    monkeypatch.setattr(m.place_on_frame, 'main', lambda argv: calls.append('place') or 0)
    assert m.main([]) == 2
    assert calls == ['grasp']


def test_main_needs_an_interactive_terminal(monkeypatch):
    monkeypatch.setattr(m.sys, 'stdin', SimpleNamespace(isatty=lambda: False))
    with pytest.raises(SystemExit):
        m.main([])
