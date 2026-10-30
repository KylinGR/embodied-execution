"""逐项奖励分解诊断：看清楚在某个运动剖面下，到底是哪一项在吃奖励。

为什么需要它：
    我们一直在猜「是哪一项奖励在压制原地转向 / 直线保持」，
    但从来没实测过。改奖励六次只对两次，就是因为一直在猜。

    这个脚本直接打印每个剖面下各奖励项的时间平均，按绝对值排序。
    哪一项大，哪一项就是压制源。

用法：
    python -m diablo_training.reward_breakdown_diag --model runs/A-fix14-tilt15/selection/policy-best.zip
    python -m diablo_training.reward_breakdown_diag --model ... --stages turn forward

输出示例：
    剖面 turn（指令 vx=0.00 yaw=+0.30）
      项                      每步均值      相对forward上限
      yaw_tracking            +0.00120       0.10x
      position_hold           -0.00950     -0.79x   ← 压制源
      ...
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

DT_FALLBACK = 0.01
# forward_tracking 的上限：权重 1.2 × dt 0.01
FORWARD_CEILING = 0.012

STAGES: dict[str, tuple[list[dict[str, Any]], float]] = {
    "stand": ([{"time": 0.0, "command": [0.0, 0.0, 0.25]}], 8.0),
    "forward": ([{"time": 0.0, "command": [0.0, 0.0, 0.25]},
                 {"time": 1.0, "command": [0.5, 0.0, 0.25]}], 8.0),
    "backward": ([{"time": 0.0, "command": [0.0, 0.0, 0.25]},
                  {"time": 1.0, "command": [-0.28, 0.0, 0.25]}], 8.0),
    "turn": ([{"time": 0.0, "command": [0.0, 0.0, 0.25]},
              {"time": 1.0, "command": [0.0, 0.30, 0.25]}], 8.0),
    "turn_left": ([{"time": 0.0, "command": [0.0, 0.0, 0.25]},
                   {"time": 1.0, "command": [0.0, 0.30, 0.25]}], 8.0),
}


def build_env(resolved: dict[str, Any]):
    """按 resolved.json 构建环境（与 xingtian_diag2 一致）。"""
    import importlib

    from diablo_training.contract import environment_class

    def _resolve(name: str):
        module, _, attr = name.partition(":")
        return getattr(importlib.import_module(module), attr)

    env_cls = environment_class(resolved["environment_class"])
    params = dict(resolved["environment"])

    cfg = None
    cfg_name = resolved.get("config_class")
    if cfg_name:
        try:
            cfg = _resolve(cfg_name)(**params)
        except Exception:
            cfg = None
    if cfg is None:
        from diablo_training.env import Config

        cfg = Config(**params)

    return env_cls(cfg)


def install_profile(env, profile, ramp: float = 0.0) -> None:
    """覆写官方钩子 get_command（与 xingtian_diag2 一致）。"""

    def _at(t: float) -> np.ndarray:
        if t <= profile[0]["time"]:
            return np.asarray(profile[0]["command"], dtype=float)
        for i in range(1, len(profile)):
            a, b = profile[i - 1], profile[i]
            if t < b["time"]:
                c0 = np.asarray(a["command"], dtype=float)
                c1 = np.asarray(b["command"], dtype=float)
                if ramp <= 0.0:
                    return c0
                start = b["time"] - ramp
                frac = 0.0 if t <= start else min(1.0, (t - start) / ramp)
                return c0 + (c1 - c0) * frac
        return np.asarray(profile[-1]["command"], dtype=float)

    env.get_command = lambda sim_seconds: _at(float(sim_seconds))
    env.command = _at(0.0)


def find_reward_method(env) -> Callable | None:
    """自动找"算奖励"的那个方法，不假设名字。

    优先 reward_for_transition；否则找名字含 reward / 只吃一个参数的公开方法。
    """
    for name in ("reward_for_transition", "reward", "compute_reward", "_reward"):
        fn = getattr(env, name, None)
        if callable(fn):
            return fn

    candidates = []
    for name in dir(env):
        if name.startswith("__") or "reward" not in name:
            continue
        fn = getattr(env, name, None)
        if not callable(fn):
            continue
        try:
            params = inspect.signature(fn).parameters
        except (TypeError, ValueError):
            continue
        if len(params) == 1:
            candidates.append((name, fn))
    return candidates[0][1] if candidates else None


_REGISTRY: list[Any] = []
_HOOKED = False


def install_breakdown_hook() -> list[Any]:
    """给 RewardBreakdown 类装钩子：每次构造都记下来。

    为什么不用"包装奖励方法"：第一版那样做抓不到任何东西，说明
    env.step() 根本没调用实测到的那个方法（外面可能还套了一层 env）。
    在类上装钩子与调用方无关，最稳。只装一次，重复装会重复记录。
    """
    global _HOOKED
    if _HOOKED:
        return _REGISTRY

    import diablo_training.xingtian as X

    cls = getattr(X, "RewardBreakdown", None)
    if cls is None:
        return _REGISTRY

    original = cls.__init__

    def patched(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        original(self, *args, **kwargs)
        _REGISTRY.append(self)

    cls.__init__ = patched  # type: ignore[method-assign]
    _HOOKED = True
    return _REGISTRY


def probe_env(env) -> None:
    """抓不到时用它把线索打出来。"""
    print(f"  env 类：{type(env).__name__}  模块：{type(env).__module__}")
    names = [n for n in dir(env) if "reward" in n.lower() or "break" in n.lower()]
    print(f"  相关属性/方法：{names}")
    for n in names:
        try:
            v = getattr(env, n)
        except Exception:
            continue
        if hasattr(v, "terms"):
            print(f"  找到 RewardBreakdown 实例：env.{n} = {getattr(v, 'terms', None)}")


def run_stage(env, model, name: str, seed: int) -> dict[str, Any] | None:
    profile, duration = STAGES[name]
    install_profile(env, profile)

    need_s = duration + 1.0
    try:
        current = float(getattr(env.config, "episode_seconds", 0.0))
        if current < need_s:
            env.config.episode_seconds = need_s
    except Exception:
        pass

    obs, _ = env.reset(seed=seed, options={})
    registry = install_breakdown_hook()
    caps = len(registry)

    dt = float(getattr(env.config, "control_dt_s", DT_FALLBACK) or DT_FALLBACK)
    limit = int(round(duration / dt))
    skip = int(round(1.0 / dt))  # 前 1 秒是过渡，不计入均值

    terms_sum: dict[str, float] = {}
    n = 0
    yaw0 = yaw1 = None
    xy0 = xy1 = None

    def _yaw() -> float:
        r = np.asarray(env.data.xmat[env.base_id]).reshape(3, 3)
        return float(np.arctan2(r[1, 0], r[0, 0]))

    for i in range(limit):
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, info = env.step(action)

        if i >= skip:
            for b in registry[caps:]:
                for k, v in b.terms.items():
                    terms_sum[k] = terms_sum.get(k, 0.0) + float(v)
                n += 1
            if yaw0 is None:
                yaw0 = _yaw()
                xy0 = np.array(env.data.xpos[env.base_id][:2], dtype=float)
            yaw1 = _yaw()
            xy1 = np.array(env.data.xpos[env.base_id][:2], dtype=float)
        caps = len(registry)

        if term or trunc:
            break

    if n == 0:
        return None

    means = {k: v / n for k, v in terms_sum.items()}
    dyaw = 0.0
    dist = 0.0
    if yaw0 is not None and yaw1 is not None:
        d = yaw1 - yaw0
        dyaw = float(np.degrees((d + np.pi) % (2 * np.pi) - np.pi))
    if xy0 is not None and xy1 is not None:
        dist = float(np.linalg.norm(xy1 - xy0))

    return {
        "stage": name,
        "command": profile[-1]["command"],
        "steps": n,
        "terms": means,
        "total": float(sum(means.values())),
        "dyaw_deg": dyaw,
        "dist_m": dist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--stages", nargs="+", default=["stand", "forward", "backward", "turn"])
    ap.add_argument("--seed", type=int, default=2002)
    args = ap.parse_args()

    resolved_path = args.model.parent / "resolved.json"
    if not resolved_path.exists():
        print(f"[失败] 找不到 {resolved_path}")
        return 1
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))

    from stable_baselines3 import PPO

    model = PPO.load(str(args.model), device="cpu")
    env = build_env(resolved)

    found = find_reward_method(env)
    if found is None:
        print("[失败] 找不到奖励方法，请把 env 的方法列表贴出来：")
        print("  ", [n for n in dir(env) if "reward" in n.lower()])
        return 1

    print(f"模型 {args.model}")
    print(f"奖励方法 {found.__name__}   参照 forward 上限 {FORWARD_CEILING}/步")
    print()

    for stage in args.stages:
        if stage not in STAGES:
            print(f"[跳过] 未知剖面 {stage}")
            continue
        res = run_stage(env, model, stage, args.seed)
        if res is None:
            print(f"剖面 {stage}: 没抓到任何奖励项。线索：")
            probe_env(env)
            print()
            continue
        cmd = res["command"]
        print(f"剖面 {stage}（指令 vx={cmd[0]:+.2f} yaw={cmd[1]:+.2f}）  {res['steps']} 步")
        print(f"  {'项':<26}{'每步均值':>12}{'相对forward上限':>18}")
        for k, v in sorted(res["terms"].items(), key=lambda kv: -abs(kv[1])):
            ratio = v / FORWARD_CEILING
            flag = "  ← 压制源" if ratio <= -0.5 else ("  ← 主要项" if ratio >= 0.5 else "")
            print(f"  {k:<26}{v:>+12.5f}{ratio:>+17.2f}x{flag}")
        print(f"  {'合计':<26}{res['total']:>+12.5f}{res['total'] / FORWARD_CEILING:>+17.2f}x")
        print(f"  实测：yaw 变化 {res['dyaw_deg']:+.1f}°   位移 {res['dist_m']:.2f} m")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
