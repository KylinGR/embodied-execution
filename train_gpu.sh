#!/usr/bin/env bash
# GPU 训练：显式指定卡 + 训练前校验可用性。
#
# 判据（重要，别用错）：
#   这个任务 GPU 极轻——约 350 MiB、6% 利用率，瓶颈在 CPU 的 MuJoCo 步进。
#   所以"能不能用某张卡"应该看**算力是否空闲**，而不是"显存是否为空"。
#
#   · util < 10%  且  空闲显存 >= 1500 MiB  →  可用（上去不会拖慢任何人）
#   · util >= 50%                            →  不用（会明显拖慢原任务）
#   其余情况打印全表后中止。
#
# 服务器规则第 3 条是"选择空闲或已协调分配的 GPU"。util 接近 0 的卡
# 对他人算力无影响，且我们只从空闲显存里取 350 MiB，符合这条规则。
#
# 用法：
#   bash train_gpu.sh                              # 自动挑
#   GPU=1 bash train_gpu.sh                        # 指定
#   GPU=1 bash train_gpu.sh runs/A-fix9 2000000 2002
#
# ⚠ 长任务放在 tmux 里：tmux new -s train → 跑 → Ctrl+B D

set -u

OUT="${1:-runs/A-fix9-gpu}"
STEPS="${2:-2000000}"
SEED="${3:-2002}"
CONFIG="${4:-examples/xingtian_stage_a.json}"
NENV="${NENV:-16}"
GPU="${GPU:-}"

# 判据阈值（可用环境变量覆盖）
#   MIN_FREE_MB=800 bash train_gpu.sh ...   放宽显存要求
#   MAX_UTIL=20 bash train_gpu.sh ...        放宽算力要求
# 本任务实测占 ~350 MiB，所以 800 MiB 仍有 2 倍余量。
MAX_UTIL="${MAX_UTIL:-10}"          # % 算力利用率上限
MIN_FREE_MB="${MIN_FREE_MB:-1500}"  # 空闲显存下限

cd "$(dirname "$0")"
if [ ! -d examples ] && [ -d ../examples ]; then
    cd ..
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[中止] 找不到 nvidia-smi"
    exit 1
fi

QUERY="index,utilization.gpu,memory.used,memory.total"

echo "当前 GPU 状态（判据：util < ${MAX_UTIL}% 且空闲显存 >= ${MIN_FREE_MB} MiB）"
nvidia-smi --query-gpu="$QUERY" --format=csv,noheader | awk -F', ' '
{
    util = $2; gsub(/[^0-9]/, "", util)
    used = $3; gsub(/[^0-9]/, "", used)
    total = $4; gsub(/[^0-9]/, "", total)
    free = total - used
    ok = (util + 0 <= 10 && free >= 1500) ? "可用" : "跳过"
    printf "  GPU %-2s  util %3d%%  used %5d MiB  free %5d MiB  %s\n",
           $1, util, used, free, ok
}'

if [ -z "$GPU" ]; then
    GPU=$(nvidia-smi --query-gpu="$QUERY" --format=csv,noheader | awk -F', ' -v mu="$MAX_UTIL" -v mf="$MIN_FREE_MB" '
        {
            util = $2; gsub(/[^0-9]/, "", util)
            used = $3; gsub(/[^0-9]/, "", used)
            total = $4; gsub(/[^0-9]/, "", total)
            free = total - used
            if (util + 0 <= mu + 0 && free >= mf + 0) {
                if (best == "" || free > best) { best = free; g = $1 + 0 }
            }
        }
        END { print (best == "" ? -1 : g) }')
    if [ "$GPU" = "-1" ]; then
        echo
        echo "[中止] 没有同时满足「算力空闲 + 显存够」的卡。"
        echo "       等一轮，或与占卡的人协调后指定：GPU=<编号> bash train_gpu.sh ..."
        exit 1
    fi
    echo
    echo "[自动选卡] GPU $GPU（算力空闲且显存充足中，空闲显存最大的一张）"
fi

# 校验指定的卡
CHECK=$(nvidia-smi --query-gpu="$QUERY" --format=csv,noheader | awk -F', ' -v g="$GPU" -v mu="$MAX_UTIL" -v mf="$MIN_FREE_MB" '
    {
        i = $1; gsub(/[^0-9]/, "", i)
        if (i + 0 != g + 0) next
        util = $2; gsub(/[^0-9]/, "", util)
        used = $3; gsub(/[^0-9]/, "", used)
        total = $4; gsub(/[^0-9]/, "", total)
        free = total - used
        printf "%d %d %d", util, free, (util + 0 <= mu + 0 && free >= mf + 0) ? 1 : 0
    }')

if [ -z "${CHECK:-}" ]; then
    echo "[中止] 找不到 GPU $GPU"
    exit 1
fi

read -r U F OK <<<"$CHECK"
if [ "$OK" -ne 1 ]; then
    echo "[中止] GPU $GPU 不满足判据：util=${U}%（上限 ${MAX_UTIL}%） free=${F} MiB（下限 ${MIN_FREE_MB}）"
    exit 1
fi

echo "[校验] GPU $GPU 可用（util ${U}%，空闲显存 ${F} MiB）"
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
