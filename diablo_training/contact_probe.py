"""看"到底哪些几何体在跟地面接触"，以及各部件离地多高。

背景：
  - URDF 里的 geom 大多没有名字（`model.geom(i).name` 是空串），
    所以按 geom 名打印接触对只会得到一堆 ''。
    要打 body 名：`model.body(model.geom_bodyid[g]).name`。
  - `contact.dist` 在穿透时是**负数**，穿透深度 = `-dist`。
  - 解析控制器模式下策略不参与，action 给零即可。

用法：
  BALANCE_MODE=analytic LEG_MODE=pd python -m diablo_training.contact_probe \
      --resolved runs/A-fix14-tilt15/selection/resolved.json --stage stand --seconds 5
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np

os.environ.setdefault("BALANCE_MODE", "analytic")
os.environ.setdefault("LEG_MODE", "pd")

from diablo_training import balance  # noqa: E402
from diablo_training import xingtian_diag2 as D  # noqa: E402


class _ZeroPolicy:
    def predict(self, obs, deterministic=True):  # noqa: ARG002
        return np.zeros(6, dtype=np.float32), None


def _name(env, gid: int) -> str:
    geom = env.model.geom(gid).name
    body = env.model.body(env.model.geom_bodyid[gid]).name
    return f"{body}/{geom}" if geom else body


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--stage", default="stand")
    ap.add_argument("--seconds", type=float, default=None,
                    help="覆盖剖面时长；默认用剖面自己的")
    ap.add_argument("--seed", type=int, default=2002)
    args = ap.parse_args()

    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    spec = dict(D.STAGES[args.stage])
    if args.seconds:
        spec["duration"] = float(args.seconds)
    env = D.build_env(resolved)
    D.install_profile(env, spec["profile"], 0.0)
    env.config.episode_seconds = float(spec["duration"]) + 1.0
    env.reset(seed=args.seed, options={})

    dt = float(getattr(env.config, "control_dt_s", 0.01) or 0.01)
    n = int(round(spec["duration"] / dt))
    other = Counter()
    zmin: dict[str, float] = {}
    ended = 0
    for i in range(n):
        _obs, _r, term, trunc, _info = env.step(np.zeros(6))
        if term or trunc:
            ended = i + 1
            break
        d = env.data
        for j in range(d.ncon):
            c = d.contact[j]
            a, b = _name(env, int(c.geom1)), _name(env, int(c.geom2))
            if "floor" in (a, b):
                other[a if b == "floor" else b] += 1
        for bid in range(env.model.nbody):
            nm = env.model.body(bid).name
            if not nm:
                continue
            _x, _y, z = env.data.xpos[bid]
            zmin[nm] = min(zmin.get(nm, 1e9), float(z))

    d = env.data
    rot = np.asarray(d.xmat[env.base_id]).reshape(3, 3)
    pitch = np.degrees(np.arctan2(-rot[2, 0], rot[2, 2]))
    print(f"\n[{args.stage}] 步数 {ended or n}  跌倒={bool(ended and ended < n)}")
    print(f"  base 高度 {d.xpos[env.base_id][2]:.3f} m   pitch {pitch:+.2f}°   "
          f"整机高度(base+0.112) {d.xpos[env.base_id][2] + 0.112:.3f} m")
    print("  与地面有接触的 body（按接触步数占比）：")
    for nm, cnt in other.most_common():
        print(f"    {nm:<22} {cnt}/{ended or n} 步  ({cnt / max(ended or n, 1):.0%})")
    print("  各 body 最低点（m）：")
    for nm, z in sorted(zmin.items(), key=lambda kv: kv[1]):
        print(f"    {nm:<22} {z:+.4f}")
    print("  末帧接触明细：")
    for j in range(d.ncon):
        c = d.contact[j]
        a, b = _name(env, int(c.geom1)), _name(env, int(c.geom2))
        print(f"    {a:<24} <-> {b:<24} dist {c.dist:+.4f}  穿透 {-c.dist * 1000:+.2f} mm")

    # 有名字的 geom 单独列一遍：caster 这类挂在小体上的东西不属于独立 body，
    # 上面按 body 统计会漏掉，但它们离地多高直接决定"会不会擦地"。
    print("  有名字的 geom（位置 / 半径 / 离地间隙）：")
    for gid in range(env.model.ngeom):
        nm = env.model.geom(gid).name
        if not nm or nm == "floor":
            continue
        xyz = np.asarray(d.geom_xpos[gid])
        radius = float(np.max(np.asarray(env.model.geom_size[gid])[:1]))
        print(f"    {nm:<14} body={env.model.body(env.model.geom_bodyid[gid]).name:<18}"
              f" z={xyz[2]:+.4f} r={radius:.4f} 间隙={xyz[2] - radius:+.4f}")
    try:
        env.close()
    except Exception:
        pass
    _ = balance  # 只是确保模块已加载
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
