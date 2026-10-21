#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DIABLO_BIN="${DIABLO_BIN:-.venv/bin/diablo}"
ROOT="${RUN_ROOT:-runs/curriculum}"
GPU_ID="${GPU_ID:-1}"
CPU_SET="${CPU_SET:-8-23}"
N_ENVS="${N_ENVS:-16}"
GPU_FRACTION="${GPU_FRACTION:-0.5}"
STEPS_A="${STEPS_A:-2000000}"
STEPS_B="${STEPS_B:-1500000}"
STEPS_C="${STEPS_C:-1000000}"
STEPS_D="${STEPS_D:-3000000}"

if [[ -n "$CPU_SET" ]]; then
  affinity=(taskset -c "$CPU_SET")
else
  affinity=()
fi

train_stage() {
  local config="$1"
  local steps="$2"
  local output="$3"
  local warm_start="${4:-}"
  local extra=()
  if [[ -n "$warm_start" ]]; then
    extra=(--warm-start "$warm_start")
  fi
  CUDA_VISIBLE_DEVICES="$GPU_ID" \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  "${affinity[@]}" "$DIABLO_BIN" train \
    --config "$config" \
    "${extra[@]}" \
    --device cuda:0 \
    --vec-env subproc \
    --n-envs "$N_ENVS" \
    --cpu-threads 1 \
    --gpu-fraction "$GPU_FRACTION" \
    --steps "$steps" \
    --output "$output"
}

select_stage() {
  local run="$1"
  local seed="$2"
  shift 2
  "$PYTHON_BIN" -m diablo_training.select_best \
    --run "$run" \
    --seed "$seed" \
    --min-success "${MIN_SUCCESS:-1.0}" \
    --strict \
    --stages "$@"
}

mkdir -p "$ROOT"

train_stage \
  examples/xingtian_reference.json \
  "$STEPS_A" \
  "$ROOT/A"
select_stage "$ROOT/A" 2001 stand forward backward turn start_stop
best="$ROOT/A/selection/policy-best.zip"

for level in 1 2 3 4; do
  run="$ROOT/B${level}"
  train_stage \
    "examples/xingtian_stage_b_l${level}.json" \
    "$STEPS_B" \
    "$run" \
    "$best"
  select_stage "$run" $((2001 + level)) forward backward turn start_stop
  best="$run/selection/policy-best.zip"
done

for level in 1 2 3 4; do
  run="$ROOT/C${level}"
  train_stage \
    "examples/xingtian_stage_c_l${level}.json" \
    "$STEPS_C" \
    "$run" \
    "$best"
  select_stage "$run" $((2101 + level)) forward turn start_stop
  best="$run/selection/policy-best.zip"
done

train_stage \
  examples/xingtian_stage_d_mixed.json \
  "$STEPS_D" \
  "$ROOT/D" \
  "$best"
select_stage "$ROOT/D" 2201 stand forward backward turn start_stop

printf 'Curriculum complete. Best policy: %s\n' "$ROOT/D/selection/policy-best.zip"
