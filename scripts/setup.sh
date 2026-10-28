#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

python_bin="${PYTHON:-python3}"
"$python_bin" - <<'PY'
import sys
if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
    raise SystemExit("Use Python 3.11 or 3.12")
PY

if [[ "$(uname -s)" != "Linux" ]]; then
  printf 'Training requires Linux. Windows is supported only for source inspection.\n' >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  printf 'nvidia-smi is required for MJLab training.\n' >&2
  exit 1
fi

if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
torch_index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
.venv/bin/python -m pip install \
  'torch>=2.8,<3' --index-url "$torch_index"
.venv/bin/python -m pip install -e .
.venv/bin/diablo fetch
.venv/bin/diablo doctor
.venv/bin/diablo gpu-doctor

printf '\nReady. Activate with: source .venv/bin/activate\n'
