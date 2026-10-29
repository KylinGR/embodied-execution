"""Run the normal CLI against the single-jump checkpoint's frozen source snapshot."""
from pathlib import Path
import sys

snapshot = Path(__file__).resolve().parents[1] / 'artifacts/single-jump-v1/source_snapshot'
if not (snapshot / 'diablo_training/__init__.py').is_file():
    raise RuntimeError('Missing complete single-jump source snapshot')
sys.path[:0] = [str(snapshot / 'examples'), str(snapshot)]
from diablo_training.cli import main

if __name__ == '__main__':
    main()
