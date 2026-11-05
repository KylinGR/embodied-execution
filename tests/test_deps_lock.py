"""依赖文件版本锁：改了 balance.py 却忘了重新钉版本，这里就失败。

配套 scripts/pin_deps.py。教程要求"其他依赖文件需你自行版本管理"，
这条测试就是那句话的可执行版本。
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "diablo_training" / "deps.lock.json"


class TestDepsLock(unittest.TestCase):
    def test_lock_file_exists(self):
        self.assertTrue(
            LOCK.exists(),
            "缺少 diablo_training/deps.lock.json；先运行 python scripts/pin_deps.py",
        )

    def test_locked_files_match(self):
        lock = json.loads(LOCK.read_text(encoding="utf-8"))
        self.assertIsInstance(lock, dict)
        self.assertTrue(lock, "锁文件为空")
        for rel, expected in lock.items():
            path = ROOT / rel
            self.assertTrue(path.exists(), f"锁文件提到的 {rel} 不存在")
            actual = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(
                actual,
                expected,
                f"{rel} 的内容变了但锁没更新。"
                "如果是有意改动：运行 python scripts/pin_deps.py 并提交锁文件；"
                "如果是无意改动：git checkout -- " + rel,
            )


if __name__ == "__main__":
    unittest.main()
