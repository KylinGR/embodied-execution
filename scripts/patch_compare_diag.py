"""给 compare_diag.py 加 --ignore-missing：忽略"单边缺失"的条目。

背景
----
`compare_diag.py` 会把阈值表里"新版本多出来的键"报成差异，例如：

    ❌ 发现 2 处差异（阈值 1e-06）：
       thresholds/forward_fast_08   缺失
       thresholds/forward_fast_10   缺失

这在新旧 diag.json 由不同代码版本产出时会误报——指标其实逐位一致，
只是后来加了两个剖面。独立复核时看到这个会以为结果不一致。

用法
----
    python scripts/patch_compare_diag.py
    python scripts/compare_diag.py --ignore-missing A/diag.json B/diag.json
"""

from __future__ import annotations

import ast
from pathlib import Path

TARGET = Path(__file__).resolve().parents[1] / "scripts" / "compare_diag.py"

OLD_MAIN = """def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    pa, pb = Path(sys.argv[1]), Path(sys.argv[2])
"""

NEW_MAIN = """def main() -> int:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    ignore_missing = "--ignore-missing" in sys.argv
    if len(argv) != 2:
        print(__doc__)
        return 2
    pa, pb = Path(argv[0]), Path(argv[1])
"""

OLD_TAIL = """    rows = len(ja.get("rows", []))
    if not bad:
"""

NEW_TAIL = """    if ignore_missing:
        bad = [item for item in bad if "缺失" not in str(item[1])]

    rows = len(ja.get("rows", []))
    if not bad:
"""


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")

    if "--ignore-missing" in text:
        print("[跳过] 已经加过 --ignore-missing")
        return 0

    for name, anchor in (("main 头", OLD_MAIN), ("main 尾", OLD_TAIL)):
        count = text.count(anchor)
        if count != 1:
            print(f"[失败] {name} 锚点匹配到 {count} 处，期望 1 处")
            return 1

    text = text.replace(OLD_MAIN, NEW_MAIN, 1)
    text = text.replace(OLD_TAIL, NEW_TAIL, 1)

    try:
        ast.parse(text)
    except SyntaxError as exc:
        print(f"[失败] 语法不通：{exc}")
        return 1

    TARGET.write_text(text, encoding="utf-8")
    print("[完成] compare_diag.py 支持 --ignore-missing")
    print("       用法：python scripts/compare_diag.py --ignore-missing A/diag.json B/diag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
