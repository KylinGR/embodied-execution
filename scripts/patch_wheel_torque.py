"""给仓库加一个动作模式 wheel_torque：腿用 PD 固定，轮子直接给力矩。

为什么需要：
    - position_velocity 模式里轮子力矩上限只有 ±3 N·m（τ = -3·act - 0.3·dq），
      装不下解析控制器制动时要的 12~17 N·m（实测 5.4% 的样本饱和、
      轮子最大误差 0.96 N·m，teacher 自己都跑不满 2.8 s）。
    - torque 模式有全权限，但腿没有 PD，学生必须连腿的弹簧一起学，
      BC 误差 0.6 N·m 就足以让机身塌下去。
    - 这只机器人最自然的接口是"腿 PD + 轮力矩"。

改动（两处，都是纯增量）：
    control.py  : decode_action 增加一个 elif 分支
    env.py      : action_mode 的合法值列表加一项

用法：
    python scripts/patch_wheel_torque.py            # 打补丁
    python scripts/patch_wheel_torque.py --check    # 只看当前状态
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "diablo_training" / "control.py"
ENV = ROOT / "diablo_training" / "env.py"

ANCHOR = "        requested = action * STALL_TORQUE * OBSERVATION_SIGNS\n"

BRANCH = '''        requested = action * STALL_TORQUE * OBSERVATION_SIGNS
    elif mode == "wheel_torque":
        # 腿：和解析控制器同一套 PD（kp=60, kd=2）固定到 DEFAULT_Q。
        # 注意不能用 pv 模式的 15/18——那套软 4 倍，托不住机身（实测 132 步就摔）。
        # 这里用显式下标，避免再改 control.py 的 import。
        requested = np.zeros(6)
        for _j in (0, 1, 3, 4):
            requested[_j] = 60.0 * (DEFAULT_Q[_j] - motor_q[_j]) - 2.0 * motor_dq[_j]
        # 轮：动作直接映射成力矩，全权限（±STALL_TORQUE）
        requested[WHEELS] = action[WHEELS] * STALL_TORQUE * OBSERVATION_SIGNS[WHEELS]
'''

ENV_OLD = 'if self.action_mode not in ("position_velocity", "torque"):'
ENV_NEW = 'if self.action_mode not in ("position_velocity", "torque", "wheel_torque"):'


def status() -> tuple[bool, bool]:
    control_done = "wheel_torque" in CONTROL.read_text(encoding="utf-8")
    env_done = "wheel_torque" in ENV.read_text(encoding="utf-8")
    return control_done, env_done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    control_done, env_done = status()
    print(f"control.py: {'已含 wheel_torque' if control_done else '未打补丁'}")
    print(f"env.py    : {'已含 wheel_torque' if env_done else '未打补丁'}")
    if args.check:
        return 0 if (control_done and env_done) else 1

    text = CONTROL.read_text(encoding="utf-8")
    want_head = BRANCH.rstrip("\n")
    if control_done:
        # 已打过补丁：检查分支内容是否就是期望版本，不是就整块替换
        start = text.find('    elif mode == "wheel_torque":')
        end = text.find('    elif mode == "position_velocity":', start)
        if start < 0 or end < 0:
            print("[失败] 找不到已有的 wheel_torque 分支块，请手工检查 control.py")
            return 1
        current = text[text.rfind(ANCHOR, 0, start):end].rstrip("\n")
        if current == want_head:
            print("[跳过] control.py 的 wheel_torque 分支已是最新")
        else:
            text = text[:text.rfind(ANCHOR, 0, start)] + want_head + "\n" + text[end:]
            try:
                ast.parse(text)
            except SyntaxError as exc:
                print(f"[失败] control.py 语法不通：{exc}")
                return 1
            CONTROL.write_text(text, encoding="utf-8")
            print("[完成] control.py 的 wheel_torque 分支已更新为强 PD 版本")
    else:
        if text.count(ANCHOR) != 1:
            print(f"[失败] control.py 里匹配到 {text.count(ANCHOR)} 处锚点，期望 1 处")
            return 1
        text = text.replace(ANCHOR, BRANCH, 1)
        try:
            ast.parse(text)
        except SyntaxError as exc:
            print(f"[失败] control.py 语法不通：{exc}")
            return 1
        CONTROL.write_text(text, encoding="utf-8")
        print("[完成] control.py 增加 wheel_torque 分支")

    text = ENV.read_text(encoding="utf-8")
    if not env_done:
        if text.count(ENV_OLD) != 1:
            print(f"[失败] env.py 里匹配到 {text.count(ENV_OLD)} 处，期望 1 处")
            return 1
        text = text.replace(ENV_OLD, ENV_NEW, 1)
        try:
            ast.parse(text)
        except SyntaxError as exc:
            print(f"[失败] env.py 语法不通：{exc}")
            return 1
        ENV.write_text(text, encoding="utf-8")
        print("[完成] env.py 合法动作模式加 wheel_torque")

    print("\n注意：这属于共享接口改动（action_mode 多了一个值），")
    print("      会让带 action_mode 的旧指纹失效，需要按流程评审。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
