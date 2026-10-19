#!/usr/bin/env bash
# Short real-physics acceptance chain; no claim that the resulting policy can walk.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-.venv/bin/python}"
mkdir -p runs
run_dir="$(mktemp -d "$PWD/runs/check-$(date -u +%Y%m%dT%H%M%S)-XXXXXX")"
"$python_bin" -m unittest discover -s tests -v > "$run_dir/tests.log" 2>&1
"$python_bin" -m diablo_training.cli doctor > "$run_dir/doctor.json"
"$python_bin" -m diablo_training.cli smoke --steps 200 --output "$run_dir/smoke" > "$run_dir/smoke.log"
"$python_bin" -m diablo_training.cli train --steps 2048 --n-envs 4 --output "$run_dir/train" > "$run_dir/train.log"
"$python_bin" -m diablo_training.cli train --resume "$run_dir/train/policy.zip" --steps 512 --n-envs 4 --output "$run_dir/resume" > "$run_dir/resume.log"
evaluation_status=0
"$python_bin" -m diablo_training.cli evaluate --model "$run_dir/resume/policy.zip" --output "$run_dir/evaluate" > "$run_dir/evaluate.log" || evaluation_status=$?
if [[ "$evaluation_status" != 0 && "$evaluation_status" != 2 ]]; then
  printf 'Evaluation infrastructure failed; evidence: %s\n' "$run_dir" >&2
  exit "$evaluation_status"
fi
"$python_bin" -m diablo_training.cli export --model "$run_dir/resume/policy.zip" --observations "$run_dir/evaluate/trajectory.npz" --output "$run_dir/export" > "$run_dir/export.log"
"$python_bin" - "$run_dir" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
training = json.loads((root / 'train/result.json').read_text())
resume = json.loads((root / 'resume/result.json').read_text())
assert training['parameters_changed'] and resume['parameters_changed']
assert resume['actual_total_timesteps'] > training['actual_total_timesteps']
assert list((root / 'resume/tensorboard').glob('*/*')), 'Resume must log into its new directory'
print(f'Environment/train/resume/evaluate/export chain passed: {root}')
print('Inspect evaluate/result.json for motion failures; short PPO runs are not locomotion qualification.')
PY
