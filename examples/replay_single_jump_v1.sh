#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export DIABLO_ASSET_DIR="${DIABLO_ASSET_DIR:-/home/qin/.local/share/embodied-execution}"
snapshot="$PWD/artifacts/single-jump-v1/source_snapshot"
export PYTHONPATH="$snapshot/examples:$snapshot:$PWD/examples:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
jump_python="${DIABLO_PYTHON:-/home/qin/.venvs/embodied-execution-gpu/bin/python}"
"$jump_python" "$snapshot/examples/qualify_jump.py" --model artifacts/single-jump-v1/train/policy.zip --seeds 30 --seed-start 21000 --video --output "${1:-runs/single-jump-replay-$(date +%Y%m%d-%H%M%S)}"
