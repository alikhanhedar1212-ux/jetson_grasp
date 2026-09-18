#!/usr/bin/env bash
set -euo pipefail

# 完整抓放：固定点 → 识别红块 → go → 接近 → 夹紧 → 抬升 → 保持里输入 place
#           → 回固定点 → 识别黑框 → go → 到框上方 → 松爪 → 抬离 → home 回初始位置
# 抓取段参数就是 2026-09-18 现场确认的那一套（同 ./run_grasp.sh）；
# 放置段（物块放进黑框）用 2026-09-14 现场参数。
# 用法：在项目任意目录执行 `./run_pick_and_place.sh`
# 需要先激活环境并拉起 CAN：
#   conda activate yolo_grasp
#   sudo ip link set can1 up type can bitrate 1000000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# 抓取段：--grasp-clearance-mm 15 停在顶面上方 15 mm 闭爪；抬到上方 100 mm；
#         基座系微调 -43 -28 0；90°/200 mm 包络。
# 放置段：--place-clearance-mm 55 停在黑框平面上方 55 mm；
#         松爪后沿基座 +Z 抬离 30 mm；J5 ≤60°、工具轴倾斜 ≤25°；
#         放置腿自己的目标微调默认 5 5 0（可用 --place-target-offset-mm 覆盖）。
exec python src/yolo_grasp/pick_and_place.py \
  --channel can1 \
  --grasp-clearance-mm 15 \
  --tracking-lock-height-mm 45 \
  --grasp-target-offset-mm -43 -28 0 \
  --grip-force-n 2 \
  --lift-to-mm 100 \
  --grasp-max-joint-delta-deg 90 \
  --grasp-max-travel-mm 200 \
  --target-move-mm 50 \
  --max-replans 5 \
  --place-clearance-mm 55 \
  --max-j5-deg 60 \
  --max-tilt-deg 25 \
  --open-force-n 2 \
  --retract-mm 30 \
  --device cpu \
  --imgsz 320 \
  --min-available-mb 1200
