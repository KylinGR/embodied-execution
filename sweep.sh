#!/usr/bin/env bash
# 多 seed 训练 + 自动挑权重 + 自动验收，全无人值守。
#
# ⚠ 必须在 tmux 里跑，否则 SSH 一断就收到 SIGHUP 被杀死：
#     tmux new -s train
#     bash sweep.sh
#     Ctrl+B 然后 D     # detach
#     tmux attach -t train
#
# 可断点续跑：只看 result.json 判断训练是否真的完成。
#   被中断的目录（有目录没 result.json）会删掉重训，
#   已完成的直接跳过，已有 selection/ 的不重复挑。
#
# 为什么需要多 seed：
#   同一份奖励、同一个 config，只换 seed 就出现——航向 10° vs 54°、
#   120 秒位移 2.6 m vs 12 m、跌倒 4/24 vs 0/24。
#   训练方差已经大到覆盖设计差异，单 seed 的结论不可信。
#
# 用法：
#   bash sweep.sh                          # 默认 A-fix8，2M 步，seed 2002/2003/2004
#   bash sweep.sh A-fix9 2000000 2002 2003
#   bash sweep.sh A-fix8 1000000 2002      # 先小步数试流程

set -u

PREFIX="${1:-A-fix8}"
STEPS="${2:-2000000}"
shift 2 2>/dev/null || true
SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && SEEDS=(2002 2003 2004)

# 训练入口默认用 GPU：
#   这个任务 GPU 极轻（~350 MiB / 6%），而 CPU 是大家共享的——
#   别人把线程拉满时用 CPU 会互相拖慢。慢 7 分钟换"不抢共享资源"是划算的。
# 想用 CPU：TRAIN_CMD="bash train_cpu.sh" bash sweep.sh ...
TRAIN_CMD="${TRAIN_CMD:-bash train_gpu.sh}"

cd "$(dirname "$0")"
if [ ! -d examples ] && [ -d ../examples ]; then
    cd ..
fi

START=$(date +%s)
echo "配置：前缀=$PREFIX  步数=$STEPS  seeds=${SEEDS[*]}"
echo "训练入口：$TRAIN_CMD"
echo "tmux 会话：${TMUX:-（不在 tmux 里！SSH 断开会被杀）}"
echo

for s in "${SEEDS[@]}"; do
    out="runs/${PREFIX}-s${s}"
    diag="runs/diag/${PREFIX}-s${s}"
    echo "============================================================"
    echo "[$s] 目标目录 → $out"
    echo "============================================================"

    if [ -f "$out/result.json" ]; then
        echo "[跳过训练] 已完成（result.json 存在）"
    else
        if [ -e "$out" ]; then
            case "$out" in
                runs/*) echo "[重做训练] $out 存在但没有 result.json（上次被中断），删除重训" ;;
                *) echo "[中止] 拒绝删除 runs/ 之外的路径：$out"; continue ;;
            esac
            rm -rf "$out"
        fi
        $TRAIN_CMD "$out" "$STEPS" "$s" || { echo "[失败] 训练中断"; continue; }
    fi

    if [ -f "$out/selection/policy-best.zip" ]; then
        echo "[跳过挑权重] selection/policy-best.zip 已存在"
    else
        echo "[$s] 挑权重"
        diablo select-best --run "$out" || { echo "[失败] select-best 无合格 checkpoint"; continue; }
    fi

    if [ -f "$diag/diag.json" ]; then
        echo "[跳过验收] $diag/diag.json 已存在"
    else
        echo "[$s] 验收 → $diag"
        # diag 有阶段不通过时返回 1，那是设计如此，不能中断
        python -m diablo_training.xingtian_diag2 \
            --model "$out/selection/policy-best.zip" \
            --seeds 2002 2003 2004 \
            --out "$diag" || true
    fi

    echo
done

ELAPSED=$(( $(date +%s) - START ))
echo "============================================================"
printf "全部完成，用时 %d 分 %d 秒\n" $((ELAPSED / 60)) $((ELAPSED % 60))
echo
echo "对照命令："
FILES=""
for s in "${SEEDS[@]}"; do
    f="runs/diag/${PREFIX}-s${s}/diag.json"
    [ -f "$f" ] && FILES="$FILES $f"
done
echo "  python -m diablo_training.diag_compare$FILES"
