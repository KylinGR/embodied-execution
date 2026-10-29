#!/usr/bin/env python3
"""Select the best checkpoint in one run directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from diablo_training.mjlab_tasks import select_policy_best


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="flat_balance")
    parser.add_argument("--run-dir", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    result = select_policy_best(
        args.run_dir.resolve(),
        device=args.device,
        stage_name=args.stage,
    )
    print(
        json.dumps(
            {
                "bestmodel_pt": result["bestmodel_pt"],
                "bestmodel_onnx": result["bestmodel_onnx"],
                "selection": str(
                    Path(result["bestmodel_pt"]).parent / "selection.json"
                ),
                "score": result["selected"]["score"],
                "blind_passed": result["blind"]["passed"],
                "limit_passed": result["limit"]["passed"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0 if result["blind"]["passed"] and result["limit"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
