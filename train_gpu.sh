#!/usr/bin/env bash
# GPU 训练：显式指定卡 + 训练前校验该卡空闲。
#
# 为什么改用 GPU：
#   之前用 CPU 是为了避开 GPU 排队，但 CPU 是大家共享的——
#   别人把线程拉满时，训练会变慢，而且我们在抢别人的资源。
#   这个任务 GPU 利用率只有 ~6%，占一张空闲卡对别人几乎无影响。
#
# 服务器规则第 3 条：训练前查 nvidia-smi，只用空闲或已协调分配的卡。
# 这个脚本把这条规则做成硬检查：目标卡显存 >= 1024 MiB 就中止。
#
# 用法：
#   bash train_gpu.sh                          # 自动挑显存最少的卡
#   GPU=2 bash train_gpu.sh                    # 指定 2 号卡
#   GPU=2 bash train_gpu.sh runs/A-fix9 2000000 2002
#
# ⚠ 长任务请放在 tmux 里：
#   tmux new -s train   →   bash train_gpu.sh   →   Ctrl+B 然后 D

set -u

OUT="${1:-runs/A-fix9-gpu}"
STEPS="${2:-2000000}"
SEED="${3:-2002}"
CONFIG="${4:-examples/xingtian_stage_a.json}"
NENV="${NENV:-16}"
GPU="${GPU:-}"

cd "$(dirname "$0")"
if [ ! -d examples ] && [ -d ../examples ]; then
    cd ..
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[中止] 找不到 nvidia-smi"
    exit 1
fi

if [ -z "$GPU" ]; then
    GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' '
        { m = $2; gsub(/[^0-9]/, "", m)
          if (NR == 1 || m + 0 < best) { best = m + 0; g = $1 + 0 } }
        END { print g }')
    echo "[自动选卡] 显存占用最小的卡：GPU $GPU"
fi

USED=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F', ' -v g="$GPU" '
    { i = $1; gsub(/[^0-9]/, "", i); m = $2; gsub(/[^0-9]/, "", m)
      if (i + 0 == g + 0) print m + 0 }')

if [ -z "${USED:-}" ]; then
    echo "[中止] 找不到 GPU $GPU"
    exit 1
fi

if [ "$USED" -ge 1024 ]; then
    echo "[中止] GPU $GPU 显存已占用 ${USED} MiB —— 服务器规则：只用空闲卡"
    echo "       换一张：GPU=<编号> bash train_gpu.sh ..."
    echo "       当前占用："
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader | sed 's/^/         /'
    exit 1
fi

echo "[校验] GPU $GPU 空闲（显存 ${USED} MiB）"
echo "config = $CONFIG"
echo "steps  = $STEPS   seed = $SEED   n_envs = $NENV   device = cuda:$GPU"
echo "output = $OUT"
echo

CUDA_VISIBLE_DEVICES="$GPU" diablo train \
    --config "$CONFIG" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --n-envs "$NENV" \
    --device cuda \
    --output "$OUT"
