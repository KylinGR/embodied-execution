#!/usr/bin/env bash
# 用 CPU 训练，完全不占用 GPU。
#
# 为什么用 CPU：
#   网络只有 MLP [128,64]（约 25k 参数），GPU 占用 347 MiB / 6% 利用率。
#   瓶颈是 MuJoCo 步进和 PPO 的小批量更新，都不吃 GPU。
#   实测 CPU 2073 步/秒，比 GPU 的 1496 步/秒还快。
#   CUDA_VISIBLE_DEVICES="" 把 GPU 彻底屏蔽，物理上不可能碰到别人的卡。
#
# 用法：
#   bash train_cpu.sh                       # 默认 runs/A-fix3-2m，2M 步，seed 2002
#   bash train_cpu.sh runs/A-fix4-2m 2000000 2002
#   bash train_cpu.sh runs/gate-100k 100000 2002

set -eu

OUT="${1:-runs/A-fix3-2m}"
STEPS="${2:-2000000}"
SEED="${3:-2002}"
CONFIG="${4:-examples/xingtian_stage_a.json}"
NENV="${NENV:-16}"

cd "$(dirname "$0")"
if [ ! -d examples ] && [ -d ../examples ]; then
    cd ..
fi

if [ -e "$OUT" ]; then
    echo "输出目录已存在：$OUT"
    echo "训练用 fresh_directory(exist_ok=False)，换个名字或先删掉。"
    exit 1
fi

echo "config = $CONFIG"
echo "steps  = $STEPS   seed = $SEED   n_envs = $NENV   device = cpu"
echo "output = $OUT"
echo

CUDA_VISIBLE_DEVICES="" diablo train \
    --config "$CONFIG" \
    --steps "$STEPS" \
    --seed "$SEED" \
    --n-envs "$NENV" \
    --device cpu \
    --output "$OUT"
