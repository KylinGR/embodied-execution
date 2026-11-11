"""把解析平衡控制器接到 env.py（在已有俯仰补偿补丁之上替换）。

前置条件：env.py 已经有俯仰补偿补丁（patch_pitch + patch_pitch_gate），
          即存在 `self._update_pitch_control()` 调用和 `self._pitch_bias`。

改动（3 处 + 2 个新方法）：
    · 调用行换成 _update_balance_control
    · ctrl 写入行换成 _apply_control(applied) + _balance_bias
    · 在 def step 前插入 _apply_control / _update_balance_control

用环境变量切模式，避免改 config schema：
    BALANCE_MODE=analytic   启用解析平衡控制器（默认 none，保持原行为）
    LEG_MODE=pd             腿用 PD 固定在 DEFAULT_Q，轮子完全交给解析控制器

用法：
    python -m diablo_training.patch_balance
    BALANCE_MODE=analytic LEG_MODE=pd python -m diablo_training.xingtian_diag2 ...
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DEFAULT = Path("diablo_training/env.py")

CALL_OLD = "        self._update_pitch_control()\n"
CALL_NEW = "        self._update_balance_control()\n"

CTRL_OLD = "            self.data.ctrl[:] = applied + self._pitch_bias\n"
CTRL_NEW = "            self.data.ctrl[:] = self._apply_control(applied) + self._balance_bias\n"

STEP_OLD = "    def step(self, action):\n"
STEP_NEW = '''    def _apply_control(self, applied):
        """LEG_MODE=pd 时腿由 PD 固定、轮子完全交给解析控制器。"""
        import os

        if os.environ.get("LEG_MODE", "policy") != "pd":
            return applied

        out = np.array(applied, dtype=float, copy=True)
        q = self.data.qpos[self.qi]
        dq = self.data.qvel[self.vi]
        for j in (0, 1, 3, 4):
            out[j] = 60.0 * (DEFAULT_Q[j] - q[j]) - 2.0 * dq[j]
        out[2] = 0.0
        out[5] = 0.0
        return out

    def _update_balance_control(self):
        """解析平衡/速度/航向控制器，BALANCE_MODE=analytic 时生效。"""
        import os

        from . import balance

        if os.environ.get("BALANCE_MODE", "none") != "analytic":
            self._balance_bias = np.zeros(6)
            return

        if getattr(self, "_bal_state", None) is None or getattr(self, "steps", 0) == 0:
            self._bal_state = balance.make_state(self.data, self.base_id)

        self._balance_bias = balance.compute(
            self._bal_state,
            self.data,
            self.model,
            self.base_id,
            self.command,
            CONTROL_DT,
        )

    def step(self, action):
'''


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[失败] 找不到 {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    if "_update_balance_control" in text:
        print("[跳过] 已经打过解析平衡补丁")
        return 0

    for name, old in (("调用行", CALL_OLD), ("ctrl 行", CTRL_OLD),
                      ("step 定义", STEP_OLD)):
        n = text.count(old)
        if n != 1:
            print(f"[失败] {name} 期望匹配 1 处，实际 {n} 处 —— 先确认俯仰补偿补丁已打")
            return 1

    new = text
    new = new.replace(CALL_OLD, CALL_NEW, 1)
    new = new.replace(CTRL_OLD, CTRL_NEW, 1)
    new = new.replace(STEP_OLD, STEP_NEW, 1)

    try:
        ast.parse(new)
    except SyntaxError as exc:
        print(f"[失败] 语法不通：{exc}")
        return 1

    path.write_text(new, encoding="utf-8")
    print(f"[完成] {path} 已接入解析平衡控制器")
    print("       用法：BALANCE_MODE=analytic LEG_MODE=pd python -m ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
