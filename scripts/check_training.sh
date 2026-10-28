#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-.venv/bin/python}"
mkdir -p runs
run_dir="$(mktemp -d "$PWD/runs/check-$(date -u +%Y%m%dT%H%M%S)-XXXXXX")"

"$python_bin" -m compileall -q diablo_training tests
"$python_bin" -m unittest discover -s tests -v 2>&1 | tee "$run_dir/tests.log"
"$python_bin" -m diablo_training.cli doctor > "$run_dir/doctor.json"
"$python_bin" -m diablo_training.contract > "$run_dir/course-contract.json"

if [[ "${DIABLO_GPU_CHECK:-0}" == "1" ]]; then
  "$python_bin" -m diablo_training.cli gpu-doctor > "$run_dir/gpu-doctor.json"
  "$python_bin" -m diablo_training.cli smoke \
    --stage flat_balance --n-envs 2 --steps 10 \
    --output "$run_dir/smoke" > "$run_dir/smoke.log"
  "$python_bin" -m diablo_training.cli train \
    --stage flat_balance --n-envs 2 --iterations 1 \
    --log-root "$run_dir/logs/rsl_rl" > "$run_dir/train.log"
  checkpoint="$(find "$run_dir/logs/rsl_rl/diablo_xingtian_course" \
    -type f -name 'model_*.pt' | sort | tail -n 1)"
  test -n "$checkpoint"
  evaluation_status=0
  "$python_bin" -m diablo_training.cli evaluate \
    --stage flat_balance --checkpoint "$checkpoint" --seeds 2001 \
    --output "$run_dir/evaluate" > "$run_dir/evaluate.log" || evaluation_status=$?
  if [[ "$evaluation_status" != 0 && "$evaluation_status" != 2 ]]; then
    printf 'GPU evaluation infrastructure failed; evidence: %s\n' "$run_dir" >&2
    exit "$evaluation_status"
  fi
  "$python_bin" -m diablo_training.cli export \
    --stage flat_balance --checkpoint "$checkpoint" \
    --output "$run_dir/export" > "$run_dir/export.log"
  test -s "$run_dir/export/policy.onnx"
else
  printf 'GPU checks skipped. Set DIABLO_GPU_CHECK=1 on a CUDA Linux host.\n'
fi

printf 'Checks completed. Evidence: %s\n' "$run_dir"
