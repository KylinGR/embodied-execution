#!/usr/bin/env bash
# 训练前环境检查：挑空闲 GPU、看 CPU 争用、给出可直接复制的训练命令。
#
# 服务器使用规则（本项目约定）：
#   · 只使用「显存占用接近 0」的 GPU，不挤占别人正在跑的卡
#   · 训练前必须显式指定 CUDA_VISIBLE_DEVICES，不用默认的 0 号卡
#   · 这个任务 GPU 利用率只有 ~6%，瓶颈是 CPU 上的 MuJoCo 步进，
#     所以 n_envs 要按「空闲核数」定，不是按 GPU 数量
#
# 用法：
#   bash check_env.sh
#   bash check_env.sh examples/xingtian_stage_b_l1.json     # 换 config

set -u

CONFIG="${1:-examples/xingtian_stage_a.json}"
GPU_QUERY="index,utilization.gpu,memory.used,memory.total"

echo "==================== GPU ===================="
nvidia-smi --query-gpu="$GPU_QUERY" --format=csv,noheader | awk -F', ' '
{
    util = $2; gsub(/[^0-9]/, "", util)
    used = $3; gsub(/[^0-9]/, "", used)
    total = $4; gsub(/[^0-9]/, "", total)
    state = (used + 0 < 1024 && util + 0 < 10) ? "空闲" : "占用"
    printf "  GPU %-2s  util %3d%%  mem %5d/%-5d MiB  %s\n",
           $1, util, used, total, state
}'

GPU=$(nvidia-smi --query-gpu="$GPU_QUERY" --format=csv,noheader | awk -F', ' '
{
    used = $3; gsub(/[^0-9]/, "", used)
    if (NR == 1 || used + 0 < best) { best = used + 0; pick = $1 + 0 }
}
END { print (best < 1024) ? pick : -1 }')

echo
if [ "$GPU" -ge 0 ]; then
    echo "  → 建议使用：CUDA_VISIBLE_DEVICES=$GPU"
else
    echo "  → 没有空闲 GPU（判据：显存 < 1024 MiB 且 util < 10%）"
    echo "    等一轮，或与占卡的人协调"
fi

echo
echo "==================== 占卡进程 ===================="
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader | \
while IFS=, read -r _uuid pid mem; do
    pid=$(printf '%s' "$pid" | tr -d ' ')
    user=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ -z "$user" ] && user="?"
    printf "  pid %-8s user %-14s %s MiB\n" "$pid" "$user" \
           "$(printf '%s' "$mem" | sed 's/ MiB//')"
done

echo
echo "==================== CPU ===================="
CORES=$(nproc)
LOAD1=$(cut -d' ' -f1 /proc/loadavg)
printf "  逻辑核数        : %s\n" "$CORES"
printf "  负载 1/5/15 分钟: %s\n" "$(cut -d' ' -f1-3 /proc/loadavg)"
ps -eo pcpu= | awk '{s += $1} END{printf "  当前总占用      : %.0f%% (约 %.1f 个核)\n", s, s / 100}'

NENV=$(awk -v c="$CORES" -v l="$LOAD1" 'BEGIN{
    h = c - l
    if (h < 4)  h = 4
    if (h > 16) h = 16
    printf "%d", int(h)
}')
FREE=$(awk -v c="$CORES" -v l="$LOAD1" 'BEGIN{printf "%.0f", c - l}')

echo
echo "==================== 建议命令 ===================="
printf "  空闲核约 %s 个 → 建议 --n-envs %s\n\n" "$FREE" "$NENV"

if [ "$GPU" -lt 0 ]; then
    GPU_HINT="<先等一张空闲卡>"
else
    GPU_HINT="$GPU"
fi

echo "  cd /home/yuewenzhe/projects/embodied-execution && \\"
echo "  CUDA_VISIBLE_DEVICES=$GPU_HINT diablo train \\"
echo "      --config $CONFIG \\"
echo "      --steps 2000000 --seed 2002 --n-envs $NENV \\"
echo "      --output runs/A-fix1-2m"
echo
echo '  # 中途看曲线（每 5 分钟一次）：'
echo '  # awk -F, '"'"'NR>2{n++;s+=$2;if($2>m)m=$2}END{printf "回合 %d 平均 %.0f 最长 %d\n",n,s/n,m}'"'"' runs/A-fix1-2m/episodes.csv.monitor.csv'
