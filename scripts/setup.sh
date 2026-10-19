#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python_bin="${PYTHON:-python3}"
"$python_bin" -c 'import sys; assert (3,11) <= sys.version_info[:2] < (3,13), "Use Python 3.11 or 3.12 (see docs/tutorial.md)"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
if ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  .venv/bin/python -m pip install 'torch>=2.8,<3' --index-url https://download.pytorch.org/whl/cpu
fi
.venv/bin/python -m pip install -e .
.venv/bin/diablo fetch
.venv/bin/diablo doctor
printf '\nReady. Activate with: source .venv/bin/activate\n'
