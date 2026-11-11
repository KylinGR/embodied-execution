#!/usr/bin/env bash
# 自动挑一张空闲 GPU，并把命令跑在上面。
#
# 规则（都可以用环境变量改）：
#   GPU_MIN_FREE_GB  剩余显存下限，默认 18（3090 是 24 GB）
#   GPU_MAX_UTIL     利用率上限，默认 20
# 不满足条件的卡直接跳过；一张都不满足就退出，绝不抢别人的卡。
#
# 用法：
#   scripts/pick_gpu.sh --list                       # 只看卡况，不跑东西
#   scripts/pick_gpu.sh python -m diablo_training.distill_balance dagger ...
#   scripts/pick_gpu.sh --min-free-gb 10 bash scripts/dr_sweep.sh
#
# 选中的卡号写进 CUDA_VISIBLE_DEVICES，子进程里 cuda:0 就是它。

set -uo pipefail

MIN_FREE_GB=${GPU_MIN_FREE_GB:-18}
MAX_UTIL=${GPU_MAX_UTIL:-20}
LIST_ONLY=0
CPU_FALLBACK=0
ARGS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --list) LIST_ONLY=1; shift ;;
    --min-free-gb) MIN_FREE_GB=$2; shift 2 ;;
    --max-util) MAX_UTIL=$2; shift 2 ;;
    --cpu-fallback) CPU_FALLBACK=1; shift ;;
    -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[pick_gpu] 找不到 nvidia-smi，改用 CPU 执行" >&2
  exec "${ARGS[@]}"
fi

mapfile -t ROWS < <(nvidia-smi \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader,nounits 2>/dev/null)

if [ ${#ROWS[@]} -eq 0 ]; then
  echo "[pick_gpu] nvidia-smi 没返回任何卡，改用 CPU 执行" >&2
  exec "${ARGS[@]}"
fi

printf "%-4s %-10s %-10s %-8s %-8s\n" GPU USED_MB TOTAL_MB UTIL% FREE_MB
BEST_IDX=""
BEST_SCORE=999999

for ROW in "${ROWS[@]}"; do
  IFS=',' read -r IDX USED TOTAL UTIL <<< "$ROW"
  IDX=$(echo "$IDX" | tr -d ' ')
  USED=$(echo "$USED" | tr -d ' ')
  TOTAL=$(echo "$TOTAL" | tr -d ' ')
  UTIL=$(echo "$UTIL" | tr -d ' ')
  # nvidia-smi 偶尔返回 [N/A]（GPU 刚上电、MIG 等），一律当作满载
  case "$USED" in ''|*[!0-9]*) USED=0 ;; esac
  case "$TOTAL" in ''|*[!0-9]*) TOTAL=1 ;; esac
  case "$UTIL" in ''|*[!0-9]*) UTIL=100 ;; esac
  FREE=$(( TOTAL - USED ))
  printf "%-4s %-10s %-10s %-8s %-8s\n" "$IDX" "$USED" "$TOTAL" "$UTIL" "$FREE"

  if [ "$FREE" -lt $(( MIN_FREE_GB * 1024 )) ]; then continue; fi
  if [ "$UTIL" -gt "$MAX_UTIL" ]; then continue; fi

  # 打分：显存占用为主，利用率为辅
  SCORE=$(( USED * 100 / TOTAL + UTIL ))
  if [ "$SCORE" -lt "$BEST_SCORE" ]; then
    BEST_SCORE=$SCORE
    BEST_IDX=$IDX
  fi
done

if [ "$LIST_ONLY" -eq 1 ]; then
  exit 0
fi

if [ -z "$BEST_IDX" ]; then
  echo "[pick_gpu] 没有满足条件的卡（剩余显存 ≥ ${MIN_FREE_GB} GB 且利用率 ≤ ${MAX_UTIL}%）。" >&2
  if [ "$CPU_FALLBACK" -eq 1 ] && [ ${#ARGS[@]} -gt 0 ]; then
    echo "[pick_gpu] --cpu-fallback：不占显卡，直接用 CPU 执行：${ARGS[*]}" >&2
    exec env CUDA_VISIBLE_DEVICES="" "${ARGS[@]}"
  fi
  echo "[pick_gpu] 先跟卡主协调，或用 --min-free-gb / --max-util 放宽；也可以加 --cpu-fallback。" >&2
  exit 3
fi

if [ ${#ARGS[@]} -eq 0 ]; then
  echo "[pick_gpu] 选中的卡：$BEST_IDX（没有要执行的命令）"
  exit 0
fi

echo "[pick_gpu] 选中 GPU $BEST_IDX（剩余 $(( FREE / 1024 )) GB），开始执行：${ARGS[*]}"
exec env CUDA_VISIBLE_DEVICES="$BEST_IDX" "${ARGS[@]}"
