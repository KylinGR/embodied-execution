"""俯仰补偿的符号与量级标定（开环，不训练）。

为什么要先测：
    我们准备在 env 里加一个"俯仰积分补偿"，把补偿力矩加到两个轮子上。
    符号搞反 = 正反馈 = 直接把机器人推倒。所以先量清楚：
        · 两轮同时 +τ 时，机身俯仰往哪边转
        · 两轮同时 −τ 时，反过来是否对称
        · 多大的 τ 能产生多大补偿效果（量级参考）

做法：
    用真实的 env.step 管线（和训练完全一致），动作里只给两个轮子一个常值，
    腿的力矩为 0，跑 0.3 秒，记录俯仰角变化。三次试验从同一 seed reset。

用法：
    python -m diablo_training.pitch_comp_test \
        --resolved runs/A-fix14-tilt15/selection/resolved.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 2002
DURATION = 0.3
MAGNITUDES = (0.0, 0.06, 0.12, 0.24)  # 动作量级（×17 N·m ⇒ 0 / 1 / 2 / 4 N·m）


def rotation(env) -> np.ndarray:
    return np.asarray(env.data.xmat[env.base_id]).reshape(3, 3)


def pitch_deg(env) -> float:
    """俯仰角（度）。正负号用于比较方向，绝对值仅供量级参考。"""
    r = rotation(env)
    return float(np.degrees(np.arctan2(-r[2, 0], r[2, 2])))


def roll_deg(env) -> float:
    r = rotation(env)
    return float(np.degrees(np.arctan2(r[2, 1], r[2, 2])))


def trial(env, wheel_action: float, seconds: float = DURATION) -> dict[str, float]:
    obs, _ = env.reset(seed=SEED, options={})
    action = np.zeros(6, dtype=float)
    action[2] = wheel_action  # left_wheel
    action[5] = wheel_action  # right_wheel

    dt = 0.01
    steps = int(round(seconds / dt))
    trace = []
    for i in range(steps):
        obs, _r, term, trunc, _info = env.step(action)
        if (i + 1) % int(0.1 / dt) == 0:
            trace.append((round((i + 1) * dt, 2), pitch_deg(env), roll_deg(env)))
        if term or trunc:
            break
    return {
        "pitch_end": pitch_deg(env),
        "roll_end": roll_deg(env),
        "trace": trace,
    }


def closed_loop(env, seconds: float, model=None) -> None:
    """跑一段，看俯仰/偏航随时间怎么走。

    必须给 policy —— 零动作时腿没有力矩，机器人会趴下，测不出俯仰补偿。
    用训练好的策略跑，才是"站立时俯仰如何"的真实场景。
    """
    obs, _ = env.reset(seed=SEED, options={})
    dt = 0.01
    steps = int(round(seconds / dt))
    print(f"{'时间(s)':>8}{'pitch°':>10}{'roll°':>10}{'yaw°':>10}")
    print("-" * 40)
    for i in range(steps):
        if model is None:
            action = np.zeros(6, dtype=float)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _r, term, trunc, _info = env.step(action)
        if (i + 1) % int(round(0.5 / dt)) == 0:
            r = rotation(env)
            yaw = float(np.degrees(np.arctan2(r[1, 0], r[0, 0])))
            print(f"{(i + 1) * dt:>8.1f}{pitch_deg(env):>+10.2f}"
                  f"{roll_deg(env):>+10.2f}{yaw:>+10.2f}")
        if term or trunc:
            print(f"  [{i * dt:.2f}s 终止] terminated={term} truncated={trunc}")
            break


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolved", required=True, type=Path)
    ap.add_argument("--closed-loop", action="store_true",
                    help="跑一段，观察俯仰/偏航（用于验证补偿）")
    ap.add_argument("--model", type=Path, default=None,
                    help="用这个策略跑；不给则零动作（零动作会趴下，测不出补偿）")
    ap.add_argument("--command", nargs=3, type=float, default=None,
                    help="固定指令 vx yaw height；默认用环境自带调度")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()

    from diablo_training.xingtian_diag2 import build_env

    resolved = json.loads(args.resolved.read_text(encoding="utf-8"))
    env = build_env(resolved)

    if args.closed_loop:
        has = hasattr(env, "_update_pitch_control")
        print(f"俯仰补偿是否已启用：{'是' if has else '否'}")
        if args.command is not None:
            cmd = np.asarray(args.command, dtype=float)
            env.get_command = lambda sim_seconds: cmd
            env.command = cmd
            print(f"固定指令：vx={cmd[0]:+.2f} yaw={cmd[1]:+.2f} h={cmd[2]:.2f}")
        if args.model is None:
            print("⚠️ 未给 --model：零动作时腿无力矩，机器人会趴下，结果无意义")
        print()
        model = None
        if args.model is not None:
            from stable_baselines3 import PPO

            model = PPO.load(str(args.model), device="cpu")
        closed_loop(env, args.seconds, model)
        return 0

    names = [env.model.actuator(i).name for i in range(env.model.nu)]
    print(f"执行器顺序：{names}")
    print(f"假定轮子在索引 2 和 5 —— 请核对上面的名字")
    print()
    print(f"每次从同一 seed ({SEED}) reset，给两个轮子同一个常值，腿力矩为 0，跑 {DURATION}s")
    print()
    print(f"{'动作':>8}{'力矩(N·m)':>11}{'0.1s':>9}{'0.2s':>9}{'0.3s':>9}{'roll@0.3s':>11}")
    print("-" * 58)

    results = []
    for mag in MAGNITUDES:
        res = trial(env, mag)
        trace = res["trace"]
        cols = []
        for t, p, _r in trace[:3]:
            cols.append(f"{p:>+9.2f}")
        while len(cols) < 3:
            cols.append(f"{'—':>9}")
        print(f"{mag:>8.2f}{mag * 17.0:>11.1f}" + "".join(cols)
              + f"{res['roll_end']:>+11.2f}")
        results.append((mag, res["pitch_end"]))

    print("-" * 58)
    print()
    print("判读：")
    print("  · 看 pitch 随动作量级的**变化方向**：")
    print("      +值变大 → 补偿正俯仰要用正力矩；+值变小/变负 → 要用负力矩")
    print("  · 看 ±是否大致对称（对称说明线性区，可以安全加增益）")
    print("  · 看 0.2 → 0.3 秒是否已经发散（发散说明量级太大，增益要小）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
