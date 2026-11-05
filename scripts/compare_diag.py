"""对比两次 diag.json 的全部数值指标，用来做"重构前后必须逐位相同"的回归。

用法：
    python scripts/compare_diag.py runs/diag/analytic-v3/diag.json runs/diag/analytic-v4/diag.json

判据：所有数值字段的最大绝对差 < 1e-6 才算通过（浮点累加次序变了会留 1e-12 量级，
超过 1e-6 就一定是有行为改变，不是数值噪声）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

TOL = 1e-6


def walk(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((path + "/" + str(k), "缺失", None))
                continue
            walk(a[k], b[k], path + "/" + str(k), out)
        return
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"长度 {len(a)} vs {len(b)}", None))
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
        return
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        if a != b:
            out.append((path, f"{a!r} vs {b!r}", None))
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        diff = abs(float(a) - float(b))
        if diff > TOL:
            out.append((path, f"{a} vs {b}", diff))
        return
    if a != b:
        out.append((path, f"{a!r} vs {b!r}", None))


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    pa, pb = Path(sys.argv[1]), Path(sys.argv[2])
    ja = json.loads(pa.read_text(encoding="utf-8"))
    jb = json.loads(pb.read_text(encoding="utf-8"))

    bad: list = []
    walk(ja.get("rows"), jb.get("rows"), "rows", bad)

    # 阈值/剖面定义也一并比，防止"指标一样但判据被改"
    walk(ja.get("thresholds"), jb.get("thresholds"), "thresholds", bad)

    rows = len(ja.get("rows", []))
    if not bad:
        print(f"✅ 一致：{rows} 行 × 全部数值字段，最大差 < {TOL:g}")
        print(f"   {pa.parent.name}  ==  {pb.parent.name}")
        return 0

    print(f"❌ 发现 {len(bad)} 处差异（阈值 {TOL:g}）：")
    for path, desc, diff in bad[:40]:
        extra = f"  Δ={diff:.3e}" if diff is not None else ""
        print(f"   {path:<60} {desc}{extra}")
    if len(bad) > 40:
        print(f"   ... 还有 {len(bad) - 40} 处")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
