# Piper X 视觉抓放 · Jetson + RealSense D405

在 Jetson 上用一台腕部 RealSense D405，让 AgileX **Piper X** 机械臂把红色物块抓起来放进蓝垫上的黑色方框：

`固定抓取点 → 识别红块 → go → 直线接近 → 闭爪（力反馈确认）→ 抬升 → place → 识别黑框 → go → 松爪 → 抬离 → home`

> End-to-end visual pick & place for an AgileX Piper X arm on Jetson. A wrist-mounted RealSense D405
> locates a red block and a black frame on the mat; the arm grasps the block and drops it into the frame.

## 特点

- **运动全程实时检测红块**：YOLO 跑在独立子进程（≤10 fps），物块移动超过阈值就停稳、重新识别、从当前位置重新规划（默认最多 5 轮），不是"看一眼就盲走"。
- **高度参考取物块顶面**：不依赖相机"最近的那个面"，避免参考点随视角下滑。
- **末段锁定轨迹**：降到顶面上方 45 mm 后不再重规划，走到 15 mm 才闭爪，减少临接触抖动。
- **夹取成功靠夹爪反馈判定**：闭不动 + 保持确认才抬升，闭到 3 mm 直接报错不抬升。
- **放置段实时识别但不重规划**：黑框不动，路径按固定抓取点规划一次；固定点识别取 3 帧中值。
- **全程守卫**：关节限位、运动包络、跟随误差、反馈新鲜度、使能/CAN 模式、终点法兰复核，任何失败都请求电子急停。

## 环境

- Jetson + JetPack 6（实测 L4T R36.4.7 / Ubuntu 22.04），Python 3.10
- AgileX Piper X（实测固件 S-V1.8-7）、AGX 夹爪、SocketCAN `can1` @ 1 Mbps
- Intel RealSense D405（腕部）；依赖见 `requirements.txt`

```bash
conda create -n yolo_grasp python=3.10 && conda activate yolo_grasp
pip install -r requirements.txt
sudo ip link set can1 up type can bitrate 1000000
ip -details link show can1
```

## 运行

```bash
cd /path/to/this/repo
conda activate yolo_grasp

# 1) 准备（可选）：张爪并走到固定抓取点，到位后输入 quit
python src/yolo_grasp/move_to_mat.py --open-max --force-n 1

# 2) 完整抓放（抓取 + 放进黑框），等价 ./run_pick_and_place.sh
python src/yolo_grasp/pick_and_place.py --channel can1 --min-available-mb 1200
```

运行中的输入（每一步都要人工确认，程序不会自己开动）：

| 步骤 | 输入 | 动作 |
| --- | --- | --- |
| 1 | 识别到红块后 `go` | 接近 → 停在物块顶面上方 15 mm |
| 2 | 自动 | 闭爪到夹住 → 沿 +Z 抬到顶面上方 100 mm |
| 3 | 抬升后 `place` | 携物回固定抓取点 → 识别黑框 |
| 4 | 识别到黑框后 `go` | 接近到框面上方 55 mm → 松爪 → 抬离 30 mm |
| 5 | 到位后 `home` | 先回固定抓取点，再回初始位置 |

只抓取、不放置：

```bash
./run_grasp.sh          # 或 python src/yolo_grasp/hover_red_block.py --channel can1 --grasp
```

已夹着物块、只想放置：

```bash
python src/yolo_grasp/place_on_frame.py --channel can1 --min-available-mb 1200
```

任何时候 **空格 / Esc / Ctrl+C** = 请求电子急停。急停后：

```bash
python src/yolo_grasp/reset_estop.py                                   # 按提示复位
PYTHONPATH=src:src/yolo_grasp python -m grasp arm --channel can1       # 使能 / home / 失能控制台
```

## 自检（不接机械臂、不动电机）

```bash
PYTHONPATH=src:src/yolo_grasp python -m pytest src/yolo_grasp/tests -q    # 409 passed
python src/yolo_grasp/pick_and_place.py --help
```

测试会像真实运行一样在 `src/yolo_grasp/runs/` 下写记录（已在 `.gitignore` 中，可随时删除）。

## 主要参数

抓取腿（`pick_and_place.py` 的 `--grasp-*` 同名转发）：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--clearance-mm` | 15 | 停在物块顶面上方的高度，到位才闭爪 |
| `--tracking-lock-height-mm` | 45 | 降到该高度后锁定轨迹、不再视觉重规划 |
| `--target-move-mm` | 50 | 物块移动超过该值（连续 2 帧）就停稳重规划 |
| `--max-replans` | 5 | 单次接近最多重规划轮数 |
| `--target-offset-mm` | -43 -28 0 | 基座系经验补偿（Z 保持 0） |
| `--grip-force-n` | 2 | 闭爪力参数，必须按自己的硬件验证 |
| `--lift-to-mm` | 100 | 夹住后抬到顶面上方 100 mm |

放置腿：

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `--place-clearance-mm` | 55 | 黑框平面上方松爪高度 |
| `--place-target-offset-mm` | 5 5 0 | 落点基座系微调 |
| `--max-j5-deg` / `--max-tilt-deg` | 60 / 25 | J5 规划上限与工具轴最大倾斜 |
| `--retract-mm` | 30 | 松爪后沿 +Z 抬离高度（受姿态限制可能小于请求值） |
| `--path-speed-deg-s` / `--final-speed-deg-s` | 7 / 5 | 平移段 / 下降段的关节速度峰值 |

## 适配到你的设备

```bash
# 换了相机：设置设备序列号与会话名，并换掉对应的手眼标定文件
export GRASP_CAMERA_SERIAL=<你的 D405 序列号>
export GRASP_HANDEYE_SESSION=<标定会话名>

# 换了工作台：重新示教初始位置
python src/yolo_grasp/move_to_mat.py --home-only     # 按提示保存/回到初始位置
```

- 手眼外参在 `src/data/handeye/d405_02_fixed_split/candidate_result.json`，是**候选值**（`validated=false`），横向偏差靠 `--target-offset-mm` 经验补偿；它只对当前相机安装姿态成立，换支架或换相机必须重新标定。
- 初始位置在 `src/yolo_grasp/local/arm_home.json`，换工作台需要重新示教。

## 目录结构

```text
run_grasp.sh / run_pick_and_place.sh   # 启动脚本
src/                                   # 运行时根目录（代码里的 data/ models/ yolo_grasp/ 都相对它解析）
├── data/handeye/…                     # 手眼标定（候选值）
├── models/red_block_best.pt           # 红块 YOLO 权重
├── pyAgxArm/                          # AgileX 官方 Python SDK（随仓库附带，LGPL-3.0）
└── yolo_grasp/
    ├── pick_and_place.py / hover_red_block.py / place_on_frame.py
    ├── move_to_mat.py / reset_estop.py / test_hover_steps.py
    ├── local/arm_home.json  tests/  grasp/  runs/（运行输出，Git 忽略）
    └── CONTINUOUS_HOVER.md             # 现场记录：守卫阈值、速度、异常处置
```

更详细的项目说明与历史记录见 [README.zh-CN.md](README.zh-CN.md)。

## 安全须知

- 同一时刻只允许**一个**程序控制机械臂；抓放运行期间不要并行开其它运动入口。
- 程序不做整臂避障，只有关节限位、运动包络（0.8°）、跟随误差（≤1.2°）、反馈新鲜度和终点复核（3 mm / 1°）。
- 抬升后物块只靠夹持力保持；确认夹稳后再输入 `place`。
- 15 mm 抓取高度与 55 mm 松爪高度都贴近硬件余量，第一次在自己的设备上运行请手放在急停上。
- 电子急停可能伴随阻尼下沉，不等于原地保持。

## 许可

仓库内附带的 AgileX `pyAgxArm` SDK 遵循 LGPL-3.0，详见 [LICENSE](LICENSE)。
