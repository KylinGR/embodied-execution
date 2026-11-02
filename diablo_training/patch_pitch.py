"""加入「俯仰积分补偿」——传统平衡控制器 + RL 的混合（方案甲）。

为什么：
    师兄判断"直行不行是平衡没搞好"，数据支持：
        站立前倾 15°、以 4.4 °/s 持续自转、0.05 m/s 持续漂移
    这是**稳态偏置**，而策略是无记忆 MLP，没有积分作用就消不掉稳态误差。
    文献做法：轮腿双足用"传统平衡控制器 + RL 残差/叠加"
    （Hybrid Balance Control via Residual RL；PPO with an Integral Compensator）。

符号（已用开环测试标定，pitch_comp_test）：
    两轮同时加正力矩 → 俯仰角下降（单调、近似线性）
    ⇒ comp = +kp*pitch + kd*rate + ki*∫pitch，三项正号

约束：
    obs 25 维 ×5 = 125、action 6 维 —— 都不变。
    只改控制律（力矩叠加），所以 policy_contract 会变，旧权重不可用。

增益可用环境变量覆盖，不必反复打补丁：
    PITCH_KP=0.15  PITCH_KD=0.0  PITCH_KI=0.03  PITCH_CLAMP=4.0  PITCH_I_LIMIT=20.0

用法：
    python -m diablo_training.patch_pitch            # 打补丁
    python -m diablo_training.unpatch_pitch          # 回退
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DEFAULT = Path("diablo_training/env.py")

LOOP_OLD = "        for _ in range(round(CONTROL_DT / PHYSICS_DT)):\n"
LOOP_NEW = (
    "        self._update_pitch_control()\n"
    "        for _ in range(round(CONTROL_DT / PHYSICS_DT)):\n"
)

CTRL_OLD = "            self.data.ctrl[:] = applied\n"
CTRL_NEW = "            self.data.ctrl[:] = applied + self._pitch_bias\n"

RESET_OLD = "        self.history[:] = self._frame()\n"
RESET_NEW = "        self.pitch_integral = 0.0\n        self.history[:] = self._frame()\n"

STEP_OLD = "    def step(self, action):\n"
STEP_NEW = '''    def _update_pitch_control(self):
        """俯仰积分补偿：每个控制步算一次，子步循环里叠加到两个轮子上。

        符号由开环标定：两轮同时加正力矩 → 俯仰角下降，故 kp/ki 取正。
        """
        import os

        kp = float(os.environ.get("PITCH_KP", "0.15"))
        kd = float(os.environ.get("PITCH_KD", "0.0"))
        ki = float(os.environ.get("PITCH_KI", "0.03"))
        clamp = float(os.environ.get("PITCH_CLAMP", "4.0"))
        i_limit = float(os.environ.get("PITCH_I_LIMIT", "20.0"))

        rot = np.asarray(self.data.xmat[self.base_id]).reshape(3, 3)
        pitch = float(np.degrees(np.arctan2(-rot[2, 0], rot[2, 2])))
        rate = float(self.data.sensor("gyro").data[1]) * 57.29578

        integral = float(getattr(self, "pitch_integral", 0.0))
        integral = float(np.clip(integral + pitch * CONTROL_DT, -i_limit, i_limit))
        self.pitch_integral = integral

        comp = kp * pitch + kd * rate + ki * integral
        comp = float(np.clip(comp, -clamp, clamp))

        bias = np.zeros(6)
        bias[2] = comp
        bias[5] = comp
        self._pitch_bias = bias

    def step(self, action):
'''


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[失败] 找不到 {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    if "pitch_integral" in text:
        print("[跳过] 已经打过俯仰补偿补丁")
        return 0

    for name, old in (
        ("子步循环", LOOP_OLD),
        ("ctrl 写入", CTRL_OLD),
        ("reset 锚点", RESET_OLD),
        ("step 定义", STEP_OLD),
    ):
        n = text.count(old)
        if n != 1:
            print(f"[失败] {name} 期望匹配 1 处，实际 {n} 处 —— 先别打补丁")
            return 1

    new = text
    new = new.replace(LOOP_OLD, LOOP_NEW, 1)
    new = new.replace(CTRL_OLD, CTRL_NEW, 1)
    new = new.replace(RESET_OLD, RESET_NEW, 1)
    new = new.replace(STEP_OLD, STEP_NEW, 1)

    try:
        ast.parse(new)
    except SyntaxError as exc:
        print(f"[失败] 语法不通：{exc}")
        return 1

    path.write_text(new, encoding="utf-8")
    print(f"[完成] {path} 已加入俯仰积分补偿（kp=0.15 kd=0 ki=0.03 clamp=4N·m）")
    print("       注意：policy_contract 已变，旧权重不可用")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
