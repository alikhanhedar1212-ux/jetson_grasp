#!/usr/bin/env bash
set -euo pipefail

# 现场抓取参数（2026-09-18 现场确认，用户指定记住的一套值）
# 用法：在项目任意目录执行 `./run_grasp.sh`，或 `bash run_grasp.sh`
# 需要先激活环境并拉起 CAN：
#   conda activate yolo_grasp
#   sudo ip link set can1 up type can bitrate 1000000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# 参数含义：
#   --clearance-mm 15              TCP 停在红块顶面上方 15 mm 后才闭爪
#   --tracking-lock-height-mm 45   降到顶面上方 45 mm 时锁定轨迹，不再视觉重规划
#   --target-offset-mm -43 -28 0   基座系目标微调（X/Y 经验值，Z 必须保持 0）
#   --lift-to-mm（未写，默认 100）  夹住后抬到顶面上方 100 mm，即相对抬升 85 mm
exec python src/yolo_grasp/hover_red_block.py \
  --grasp \
  --clearance-mm 15 \
  --tracking-lock-height-mm 45 \
  --max-joint-delta-deg 90 \
  --max-travel-mm 200 \
  --tcp-offset-mm 0 0 80 \
  --target-offset-mm -43 -28 0 \
  --target-move-mm 50 \
  --max-replans 5 \
  --channel can1 \
  --device cpu \
  --imgsz 320 \
  --min-available-mb 1200
