"""解析控制器快速调参：单阶段 / 单 seed，可覆盖常量，可打印停车瞬时轨迹。

用途：改常数后不用跑整套 diag（8 剖面 × 3 seed，1-8 分钟），
      只跑关心的剖面，几秒钟看到停车距离 / 速度误差的变化。

必须在仓库根目录跑，并且导入了 BALANCE_MODE=analytic + LEG_MODE=pd。

用法：
  # 单次评估 start_stop
  python -m diablo_training.balance_tune --resolved runs/A-fix14-tilt15/selection/resolved.json \
      --stage start_stop

  # 改几个常数再评估（点号表示字典内的键）
  python -m diablo_training.balance_tune --resolved ... --stage start_stop \
      --set TORQUE_CLAMP=17 --set MOVE.k_v=60

  # 一次跑多组，比较停车距离
  python -m diablo_training.balance_tune --resolved ... --stage start_stop \
      --variant base: --variant clamp17:STOP_CLAMP=17 --variant clamp17_halfkth:STOP_CLAMP=17,STOP_KTH_SCALE=0.5

  # 打印停车前后逐 0.05 s 的（速度 / 俯仰 / 力矩）轨迹
  python -m diablo_training.balance_tune --resolved ... --stage start_stop --trace

判据用 diag_rejudge.RELAXED（正式验收阈值），同时打印严格阈值下的原因。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("BALANCE_MODE", "analytic")
os.environ.setdefault("LEG_MODE", "pd")

from diablo_training import balance  # noqa: E402
from diablo_training import diag_rejudge as R  # noqa: E402
from diablo_training import xingtian_diag2 as D  # noqa: E402


class _DummyPolicy:
    """解析控制下轮子由 balance 决定、腿由 PD 决定，策略输出完全不参与。

    但 run_stage 里仍然会调 model.predict()，所以给个全零的假策略即可，
    这样调参不需要加载 200 万步的权重。
    """

    def predict(self, obs, deterministic=True):  # noqa: ARG002
        return np.zeros(6, dtype=np.float32), None


def apply_params(items: list[str]) -> dict[str, float]:
    """`TORQUE_CLAMP=17` / `MOVE.k_v=60` 形式的覆盖，作用在 balance 模块上。"""
    applied: dict[str, float] = {}
    for item in items:
        if not item:
            continue
        key, _, val = item.partition("=")
        num = float(val)
        if "." in key:
            grp, name = key.split(".", 1)
            getattr(balance, grp)[name] = num
        else:
            setattr(balance, key, num)
        applied[key] = num
    return applied


def snapshot() -> dict[str, object]:
    snap: dict[str, object] = {}
    for k in ("STAND", "MOVE", "YAW", "STOP"):
        d = getattr(balance, k, None)
        if d is not None:
            snap[k] = dict(d)
    for k in ("TAU_UP", "TAU_DOWN", "STOP_THRESHOLD", "STOP_ANCHOR_EACH", "TORQUE_CLAMP"):
        if hasattr(balance, k):
            snap[k] = getattr(balance, k)
    return snap


def restore(snap: dict[str, object]) -> None:
    for k in ("TAU_UP", "TAU_DOWN", "STOP_THRESHOLD", "STOP_ANCHOR_EACH", "TORQUE_CLAMP"):
        if k in snap:
            setattr(balance, k, snap[k])
    for k in ("STAND", "MOVE", "YAW", "STOP"):
        d = getattr(balance, k, None)
        if d is not None and k in snap:
            d.clear()
            d.update(snap[k])  # type: ignore[arg-type]


def judge_relaxed(stage: str, m: dict) -> tuple[bool, list[str]]:
    old = {k: dict(v) for k, v in D.THRESHOLDS.items()}
    D.THRESHOLDS.clear()
    D.THRESHOLDS.update(R.RELAXED)
    try:
        return D.judge(stage, m)
    finally:
        D.THRESHOLDS.clear()
        D.THRESHOLDS.update(old)


def install_trace(dt: float) -> tuple[list[dict], object]:
    """在 balance.compute 外面套一层，记下每一步的输入输出。"""
    rows: list[dict] = []
    orig = balance.compute
    counter = {"i": 0}

    def wrapper(state, data, model, base_id, command, dt_):  # noqa: ANN001
        out = orig(state, data, model, base_id, command, dt_)
        th = balance._pitch(data, base_id)
        rows.append({
            "t": counter["i"] * dt,
            "cmd": float(np.asarray(command).reshape(-1)[0]),
            "vx_ref": float(state["v_cmd"]),
            "v": balance._body_vx(model, data, base_id),
            "th_deg": float(np.degrees(th)),
            "tw": float(out[2] + out[5]) / 2.0,
            "turn": float(out[2] - out[5]) / 2.0,
        })
        counter["i"] += 1
        return out

    balance.compute = wrapper
    return rows, orig


def print_all(rows: list[dict], dt: float, every: float = 0.5) -> None:
    stride = max(1, int(round(every / dt)))
    print(f"  {'t':>6} {'cmd':>6} {'vx_ref':>7} {'v_true':>7} {'pitch°':>7} {'tw':>7}")
    for i in range(0, len(rows), stride):
        r = rows[i]
        print(f"  {r['t']:>6.2f} {r['cmd']:>6.2f} {r['vx_ref']:>+7.3f} "
              f"{r['v']:>+7.3f} {r['th_deg']:>+7.2f} {r['tw']:>+7.2f}")


def print_stop_trace(rows: list[dict], dt: float, before: float = 0.6,
                     after: float = 1.2) -> None:
    cv = np.array([r["cmd"] for r in rows])
    moved = np.abs(cv) > 0.05
    stops = np.where(~moved[1:] & moved[:-1])[0] + 1
    if not stops.size:
        print("[trace] 没找到停车点")
        return
    stride = max(1, int(round(0.05 / dt)))
    for n, s in enumerate(stops, 1):
        i0 = max(0, s - int(before / dt))
        i1 = min(len(rows), s + int(after / dt))
        print(f"\n[trace] 第 {n} 次停车（指令归零于 t={rows[s]['t']:.2f}s）")
        print(f"  {'t':>6} {'cmd':>6} {'vx_ref':>7} {'v_true':>7} {'pitch°':>7} {'tw':>7}")
        for i in range(i0, i1, stride):
            r = rows[i]
            mark = "  <== 停车" if i == s else ""
            print(f"  {r['t']:>6.2f} {r['cmd']:>6.2f} {r['vx_ref']:>+7.3f} "
                  f"{r['v']:>+7.3f} {r['th_deg']:>+7.2f} {r['tw']:>+7.2f}{mark}")


def run_one(resolved: dict, stage: str, seed: int, params: list[str],
            *, trace: bool = False, ramp: float = 0.0,
            wheel_radius: float = 0.1) -> tuple[dict, list[dict], float]:
    snap = snapshot()
    apply_params(params)
    rows: list[dict] = []
    orig = None
    dt = 0.01
    env = D.build_env(resolved)
    try:
        dt = float(getattr(env.config, "control_dt_s", 0.01) or 0.01)
        if trace:
            rows, orig = install_trace(dt)
        m = D.run_stage(env, _DummyPolicy(), stage, D.STAGES[stage], seed,
                        ramp=ramp, wheel_radius=wheel_radius)
    finally:
        if orig is not None:
            balance.compute = orig
        try:
            env.close()
        except Exception:
            pass
        restore(snap)
    return m, rows, dt


def main() -> int:
    ap = argparse.ArgumentParser(description="解析控制器快速调参")
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--stage", default="start_stop")
    ap.add_argument("--seed", type=int, default=2002)
    ap.add_argument("--set", action="append", default=[], help="NAME=值 或 GROUP.KEY=值")
    ap.add_argument("--variant", action="append", default=[],
                    help="标签:参数,参数；空标签用 base:")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--trace-every", type=float, default=0.5)
    ap.add_argument("--ramp", type=float, default=0.0)
    ap.add_argument("--wheel-radius", type=float, default=0.1)
    args = ap.parse_args()

    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    variants = [("cli", list(args.set))]
    for spec in args.variant:
        label, _, payload = spec.partition(":")
        variants.append((label or "variant", [p for p in payload.split(",") if p]))

    keys = {
        "start_stop": ("rise_times_s", "stop_distances_m", "n_cycles", "speed_rel_err"),
        "turn_in_motion": ("speed_rel_err", "yaw_mae_turn", "distance_ratio",
                           "lateral_drift_m", "heading_change_deg"),
        "forward": ("speed_rel_err", "distance_ratio", "rise_time_s", "overshoot",
                    "lateral_drift_m"),
        "backward": ("speed_rel_err", "distance_ratio", "heading_change_deg"),
        "accel": ("speed_rel_err", "distance_ratio"),
        "turn": ("yaw_mae", "yaw_sign_ratio", "distance_ratio"),
        "stand": ("still_turn_dps", "still_drift_mps", "tilt_rms_deg", "tilt_max_deg"),
    }.get(args.stage, ("speed_rel_err",))

    print(f"[stage={args.stage} seed={args.seed} ramp={args.ramp}]")
    for label, params in variants:
        m, rows, dt = run_one(resolved, args.stage, args.seed, params,
                              trace=args.trace, ramp=args.ramp,
                              wheel_radius=args.wheel_radius)
        ok_r, why_r = judge_relaxed(args.stage, m)
        parts = [f"dur={m.get('duration_s', 0):.1f}s",
                 f"min_up={m.get('min_upright', float('nan')):.3f}"]
        for k in keys:
            v = m.get(k)
            if v is None:
                continue
            if isinstance(v, list):
                parts.append(f"{k}={[round(x, 4) if isinstance(x, float) else x for x in v]}")
            elif isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(f"  {label:<18} {'✅' if ok_r else '❌'}  " + "  ".join(parts))
        if not ok_r:
            print(f"  {'':<18} 原因：{'; '.join(why_r)}")
        if args.trace and rows:
            if args.stage in ("start_stop",):
                print_stop_trace(rows, dt)
            else:
                print_all(rows, dt, every=args.trace_every)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
