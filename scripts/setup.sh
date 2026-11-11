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
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    .venv/bin/python -m pip install 'torch>=2.8,<3' --index-url "$TORCH_INDEX_URL"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    # The default PyPI wheel is CUDA-enabled on Linux. Set TORCH_INDEX_URL
    # explicitly when the host driver requires a particular CUDA wheel.
    .venv/bin/python -m pip install 'torch>=2.8,<3'
  else
    .venv/bin/python -m pip install 'torch>=2.8,<3' --index-url https://download.pytorch.org/whl/cpu
  fi
fi
if command -v nvidia-smi >/dev/null 2>&1 && ! .venv/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
  printf '%s\n' \
    'WARNING: an NVIDIA GPU is visible but the installed PyTorch cannot use CUDA.' \
    'Reinstall PyTorch with the wheel matching this host, for example:' \
    '.venv/bin/python -m pip uninstall -y torch && .venv/bin/python -m pip install "torch>=2.8,<3" --index-url "$TORCH_INDEX_URL"' \
    'See docs/performance.md.' >&2
fi
.venv/bin/python -m pip install -e .
.venv/bin/diablo fetch
.venv/bin/diablo doctor
printf '\nReady. Activate with: source .venv/bin/activate\n'
