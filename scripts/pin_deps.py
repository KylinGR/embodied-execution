"""把"不进 policy_contract 指纹、但影响行为"的文件钉住版本。

为什么需要：
    教程原文——"指纹覆盖其整个模块文件……其他依赖文件需你自行版本管理。"
    `policy_contract` 只哈希 model/control/env/任务模块/xingtian*，
    控制器数学所在的 `balance.py` 不在里面。改它不会让旧 checkpoint 失效，
    但会悄悄改变行为。这个脚本把它的 sha256 写进 deps.lock.json，
    配套的 tests/test_deps_lock.py 会在文件改动后失败。

用法：
    python scripts/pin_deps.py          # 有意改动后重新钉一次，并把锁文件一起提交
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "diablo_training" / "deps.lock.json"

# 影响控制行为、但不在 policy_contract 指纹里的文件
FILES = (
    "diablo_training/balance.py",
    "diablo_training/xingtian_balance_env.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    lock: dict[str, str] = {}
    for rel in FILES:
        path = ROOT / rel
        if not path.exists():
            print(f"[跳过] 不存在：{rel}")
            continue
        lock[rel] = sha256(path)

    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for rel, digest in sorted(lock.items()):
        print(f"{digest[:16]}  {rel}")
    print(f"\n已写入 {LOCK.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
