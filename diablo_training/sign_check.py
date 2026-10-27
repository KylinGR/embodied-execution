"""符号约定检查：验证「前进」和「左转」的正负号与环境 command 的约定是否一致。

为什么需要：
  diag 实测策略以 −0.21 rad/s 匀速自转，转弯指令 +0.3 时反而转到 −0.198。
  这可能是①奖励没信号学出来的偏置，也可能是②接口符号约定反了。
  两者修法完全不同：①改奖励，②必须先改接口，否则奖励修好后策略会
  "在仿真里学对、到实机上做反"。这个脚本用开环力把②排除掉。

做法：
  把腿钉在 DEFAULT_Q（相当于人扶着），只给轮子恒定扭矩，看机身怎么动。
  · 前进：两轮同号扭矩，看 base 往 +x 还是 −x 走
  · 转向：左轮负、右轮正，看 yaw 往 +z（逆时针）还是 −z 走

用法：
  python -m diablo_training.sign_check
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from diablo_training.model import DEFAULT_Q, make_model, set_pose

LEG_JOINTS = ("left_hip_motor", "left_knee_motor",
              "right_hip_motor", "right_knee_motor")
WHEELS = ("left_wheel_motor", "right_wheel_motor")
STEPS = 150          # 0.3 s，够读出符号，又不至于倒下


def _i(x) -> int:
    return int(np.asarray(x).reshape(-1)[0])


def _f(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


def _yaw(m, d) -> float:
    bid = _i(m.jnt_bodyid[_i(m.joint("floating_base").id)])
    rot = np.asarray(d.xmat[bid], dtype=float).reshape(3, 3)
    return math.atan2(rot[1, 0], rot[0, 0])


def _prep():
    """摆好 DEFAULT_Q 姿态，并把 base 放到轮子刚好着地。"""
    m = make_model()
    d = mujoco.MjData(m)
    mujoco.mj_resetData(m, d)

    a = _i(m.jnt_qposadr[_i(m.joint("floating_base").id)])
    d.qpos[a : a + 3] = [0.0, 0.0, 0.3]
    d.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]
    set_pose(m, d, list(DEFAULT_Q))
    mujoco.mj_forward(m, d)

    # 用轮子几何的下沿把机身抬到刚好接触地面
    lo = min(_f(d.geom_xpos[g][2]) - _f(m.geom_rbound[g])
             for g in range(m.ngeom)
             if "wheel" in (m.body(_i(m.geom_bodyid[g])).name or "").lower())
    d.qpos[a + 2] -= lo
    mujoco.mj_forward(m, d)
    return m, d


def run(ctrl_left: float, ctrl_right: float) -> dict[str, float]:
    m, d = _prep()
    a = _i(m.jnt_qposadr[_i(m.joint("floating_base").id)])

    # DEFAULT_Q 顺序：[left_hip, left_knee, left_wheel, right_hip, right_knee, right_wheel]
    leg_pins = []
    for name, idx in (("left_hip_motor", 0), ("left_knee_motor", 1),
                      ("right_hip_motor", 3), ("right_knee_motor", 4)):
        jid = _i(m.joint(name).id)
        leg_pins.append((_i(m.jnt_qposadr[jid]), _i(m.jnt_dofadr[jid]),
                         float(DEFAULT_Q[idx])))
    wheel_act = {n: _i(m.actuator(n + "_actuator").id) for n in WHEELS}

    x0 = _f(d.qpos[a])
    y0 = _f(d.qpos[a + 1])
    yaw0 = _yaw(m, d)

    for _ in range(STEPS):
        # 钉住腿：等价于无穷刚度位置伺服，和 leg_range 一致
        for qadr, dadr, val in leg_pins:
            d.qpos[qadr] = val
            d.qvel[dadr] = 0.0
        d.ctrl[wheel_act["left_wheel_motor"]] = ctrl_left
        d.ctrl[wheel_act["right_wheel_motor"]] = ctrl_right
        mujoco.mj_step(m, d)

    dyaw = _yaw(m, d) - yaw0
    dyaw = (dyaw + math.pi) % (2 * math.pi) - math.pi
    return {
        "dx": _f(d.qpos[a]) - x0,
        "dy": _f(d.qpos[a + 1]) - y0,
        "dyaw": dyaw,
        "z": _f(d.qpos[a + 2]),
    }


def main() -> None:
    print("=" * 68)
    print("符号约定检查（开环恒定扭矩，腿钉在 DEFAULT_Q，0.3 s）")
    print("=" * 68)

    fwd = run(+2.0, +2.0)
    bwd = run(-2.0, -2.0)
    turn_a = run(-2.0, +2.0)     # 左轮负、右轮正
    turn_b = run(+2.0, -2.0)     # 左轮正、右轮负

    print(f"\n  两轮 +2 N·m : dx={fwd['dx'] * 1000:+7.1f} mm  "
          f"dy={fwd['dy'] * 1000:+7.1f} mm  dyaw={math.degrees(fwd['dyaw']):+7.2f}°")
    print(f"  两轮 -2 N·m : dx={bwd['dx'] * 1000:+7.1f} mm  "
          f"dy={bwd['dy'] * 1000:+7.1f} mm  dyaw={math.degrees(bwd['dyaw']):+7.2f}°")
    print(f"  左- 右+     : dx={turn_a['dx'] * 1000:+7.1f} mm  "
          f"dy={turn_a['dy'] * 1000:+7.1f} mm  dyaw={math.degrees(turn_a['dyaw']):+7.2f}°")
    print(f"  左+ 右-     : dx={turn_b['dx'] * 1000:+7.1f} mm  "
          f"dy={turn_b['dy'] * 1000:+7.1f} mm  dyaw={math.degrees(turn_b['dyaw']):+7.2f}°")

    print("\n" + "=" * 68)
    print("判读")
    print("=" * 68)

    ok_fwd = fwd["dx"] > abs(bwd["dx"]) * 0.5 and fwd["dx"] > 0
    print(f"  前进符号：两轮正扭矩 dx={fwd['dx'] * 1000:+.1f} mm  "
          f"→ {'+x 为前进，与 command[0]>0 一致' if ok_fwd else '需要人工判断'}")

    d_ab = math.degrees(turn_a["dyaw"])
    d_ba = math.degrees(turn_b["dyaw"])
    print(f"  转向符号：左-右+ 得 {d_ab:+.2f}°，左+右- 得 {d_ba:+.2f}°")
    if d_ab > 0 and d_ba < 0:
        print("    → 左轮负/右轮正 = 逆时针(+z)。若 command[1]>0 表示左转，则**符号一致**")
    elif d_ab < 0 and d_ba > 0:
        print("    → 左轮负/右轮正 = 顺时针(−z)。若 command[1]>0 表示左转，则**符号相反，需要翻转**")
    else:
        print("    → 两种情况同号，说明转向不受这组扭矩控制，需人工看数据")

    print("\n  注意：本脚本用的是裸扭矩，不是策略动作；"
          "它只回答'物理上手性如何'，不回答'策略想往哪转'。")


if __name__ == "__main__":
    main()
