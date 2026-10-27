#!/usr/bin/env bash
# 多 seed 训练 + 自动挑权重 + 自动验收，全无人值守。
#
# 为什么需要多 seed：
#   同一份奖励、同一个 config，只换 seed 就出现——航向 10° vs 54°、
#   120 秒位移 2.6 m vs 12 m、跌倒 4/24 vs 0/24。
#   训练方差已经大到覆盖设计差异，单 seed 的结论不可信。
#
# 用法：
#   bash sweep.sh                          # 默认 A-fix8，2M 步，seed 2002/2003/2004
#   bash sweep.sh A-fix9 2000000 2002 2003 2004 2005
#   bash sweep.sh A-fix8 1000000 2002      # 先小步数试流程

set -u

PREFIX="${1:-A-fix8}"
STEPS="${2:-2000000}"
shift 2 2>/dev/null || true
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(2002 2003 2004)

cd "$(dirname "$0")"
if [ ! -d examples ] && [ -d ../examples ]; then
    cd ..
fi

START=$(date +%s)
echo "配置：前缀=$PREFIX  步数=$STEPS  seeds=${SEEDS[*]}"
echo

for s in "${SEEDS[@]}"; do
    out="runs/${PREFIX}-s${s}"
    diag="runs/diag/${PREFIX}-s${s}"
    echo "============================================================"
    echo "[$s] 训练 → $out"
    echo "============================================================"

    if [ -e "$out" ]; then
        echo "[跳过训练] $out 已存在"
    else
        bash train_cpu.sh "$out" "$STEPS" "$s" || { echo "[失败] 训练中断"; continue; }
    fi

    echo "[$s] 挑权重"
    diablo select-best --run "$out" || { echo "[失败] select-best 无合格 checkpoint"; continue; }

    echo "[$s] 验收 → $diag"
    # diag 有阶段不通过时返回 1，那是设计如此，不能让 set -e 中断
    python -m diablo_training.xingtian_diag2 \
        --model "$out/selection/policy-best.zip" \
        --seeds 2002 2003 2004 \
        --out "$diag" || true

    echo
done

ELAPSED=$(( $(date +%s) - START ))
echo "============================================================"
printf "全部完成，用时 %d 分 %d 秒\n" $((ELAPSED / 60)) $((ELAPSED % 60))
echo
echo "对照（把这几条一起传给 diag_compare 就能并排看）："
FILES=""
for s in "${SEEDS[@]}"; do
    f="runs/diag/${PREFIX}-s${s}/diag.json"
    [ -f "$f" ] && FILES="$FILES $f"
done
echo "  python -m diablo_training.diag_compare$FILES"
