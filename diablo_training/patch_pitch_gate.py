"""给俯仰补偿加门控：只在「指令站立」时生效。

为什么：
    A-fix33 实测：补偿让站立自转从 10.3 降到 2.49（历史最好），
    但 forward 位移比从 53% 掉到 13% —— 因为补偿器不许机身前倾，
    而前倾是轮腿机器人加速的唯一手段（重力提供前向力）。

    所以补偿要"该管的时候管，该跑的时候放"：只在 |cmd_vx| 很小时启用。
    这和 tilt_hold 的门控是同一思路。

    补偿在行驶时直接返回（积分也冻结），不影响策略的正常加速动作。

环境变量：
    PITCH_GATE_VX   指令速度门控阈值，默认 0.05 m/s；设为 99 后退化为常开

用法：
    python -m diablo_training.patch_pitch_gate
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

DEFAULT = Path("diablo_training/env.py")

ANCHOR_OLD = (
    "        rot = np.asarray(self.data.xmat[self.base_id]).reshape(3, 3)\n"
    '        pitch = float(np.degrees(np.arctan2(-rot[2, 0], rot[2, 2])))\n'
)
ANCHOR_NEW = (
    '        gate_vx = float(os.environ.get("PITCH_GATE_VX", "0.05"))\n'
    "        if abs(float(self.command[0])) >= gate_vx:\n"
    "            # 在行驶：前倾是加速手段，不能补偿；积分冻结\n"
    "            self._pitch_bias = np.zeros(6)\n"
    "            return\n"
    "        rot = np.asarray(self.data.xmat[self.base_id]).reshape(3, 3)\n"
    '        pitch = float(np.degrees(np.arctan2(-rot[2, 0], rot[2, 2])))\n'
)


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[失败] 找不到 {path}")
        return 1

    text = path.read_text(encoding="utf-8")
    if "PITCH_GATE_VX" in text:
        print("[跳过] 已经加过门控")
        return 0
    if "_update_pitch_control" not in text:
        print("[失败] 还没打俯仰补偿补丁（先跑 patch_pitch）")
        return 1

    n = text.count(ANCHOR_OLD)
    if n != 1:
        print(f"[失败] 锚点期望匹配 1 处，实际 {n} 处")
        return 1

    new = text.replace(ANCHOR_OLD, ANCHOR_NEW, 1)
    try:
        ast.parse(new)
    except SyntaxError as exc:
        print(f"[失败] 语法不通：{exc}")
        return 1

    path.write_text(new, encoding="utf-8")
    print(f"[完成] {path} 的俯仰补偿已加门控（|cmd_vx| < 0.05 才启用）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
