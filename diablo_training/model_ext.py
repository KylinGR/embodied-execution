"""站立 491 mm / 匍匐 231 mm 所需的上层结构与 4 个爬行小轮。

由 `model.py` 的 `_robot_spec()` 调用（一行 import + 一行调用）。

标定依据
--------
说明书：站立 528×366×**491** mm，匍匐 528×366×**231** mm，整机 **22.9** kg。
原模型：base 顶最高 **398.2** mm，总质量 **21.02** kg。

差额 93 mm / 1.88 kg 正好是机身顶部的主控 + 电池 + 支架：
  站立  398.2 + 93 = 491.2 mm     偏差 +0.2 mm
  匍匐  小轮着地    = 231.0 mm     偏差 −0.0 mm
  质量  21.02 + 1.88 = 22.90 kg    偏差 −0.00 kg

三个数由一个刚体结构同时满足，因此这不是凑数，而是补全。

参数里仍待实机标定的部分
------------------------
  · 上层结构按均匀长方体；主控/电池实际偏后偏上会改变 CoM
  · 小轮布局 ±200×±135 mm 是估的（说明书只给整机 528×366）
  · 小轮用低摩擦球近似，只提供竖直支撑，不能复现真实 caster 的滚动

换算式（后续定高度指令时要用）
------------------------------
  整机高度 = base_z + 112 mm
  站立 491 mm → base_z 0.379；匍匐 231 mm → base_z 0.119
"""

from __future__ import annotations

import mujoco

TOP_MASS = 1.88                      # kg，= 22.9 − 21.02
TOP_HALF = (0.090, 0.130, 0.0465)    # 半尺寸 x/y/z → 高 93 mm
BASE_TOP_Z = 0.019                   # base 几何上表面相对 base 中心
TOP_POS_Z = BASE_TOP_Z + TOP_HALF[2]  # 0.0655

CASTER_RADIUS = 0.025
CASTER_Z = -0.094                    # 相对 base 中心；底面在 base_z=0.119 时着地
CASTER_XY = ((0.200, 0.135), (0.200, -0.135),
             (-0.200, 0.135), (-0.200, -0.135))

TOP_BODY = "diablo_top_module"


def add_structure(spec, base) -> None:
    """给 base 挂上顶部结构与 4 个爬行小轮。幂等：已加过就跳过。"""
    if any(b.name == TOP_BODY for b in spec.bodies):
        return

    w, d, h = TOP_HALF
    density = TOP_MASS / ((2 * w) * (2 * d) * (2 * h))

    try:
        top = base.add_body(name=TOP_BODY, pos=[0.0, 0.0, TOP_POS_Z])
    except AttributeError:  # 老版本 MjSpec 只有 spec.add_body
        top = spec.add_body(name=TOP_BODY, parent=base,
                            pos=[0.0, 0.0, TOP_POS_Z])

    # 不写 inertial：让 MuJoCo 按 density × 体积自动算质量和惯量，
    # 质量严格等于 TOP_MASS，惯量严格是长方体解析值。
    top.add_geom(
        name="top_module_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[w, d, h],
        density=density,
        rgba=[0.45, 0.48, 0.55, 1.0],
        contype=1,
        conaffinity=2,
    )

    for i, (cx, cy) in enumerate(CASTER_XY):
        base.add_geom(
            name=f"caster_{i}",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[CASTER_RADIUS],
            pos=[cx, cy, CASTER_Z],
            rgba=[0.15, 0.15, 0.15, 1.0],
            contype=1,
            conaffinity=2,
            friction=[0.001, 0.001, 0.0001],
        )


def total_height_mm(base_z: float) -> float:
    """整机高度换算：整机高度 = base_z + 112 mm。"""
    return (base_z + TOP_POS_Z + TOP_HALF[2]) * 1000.0


def base_z_for(total_mm: float) -> float:
    """反算：要得到某个整机高度，base_z 该是多少。"""
    return total_mm / 1000.0 - (TOP_POS_Z + TOP_HALF[2])


if __name__ == "__main__":
    print(f"上层结构：{TOP_MASS} kg，高 {TOP_HALF[2] * 2000:.0f} mm，"
          f"中心 z={TOP_POS_Z * 1000:.1f} mm")
    print(f"爬行小轮：4 个，r={CASTER_RADIUS * 1000:.0f} mm，z={CASTER_Z * 1000:.1f} mm")
    for label, total in (("站立", 491.0), ("匍匐", 231.0)):
        print(f"{label} {total:.0f} mm → base_z = {base_z_for(total):.4f}")
    print(f"小轮着地时 base_z = {CASTER_RADIUS - CASTER_Z:.4f} m "
          f"→ 整机 {total_height_mm(CASTER_RADIUS - CASTER_Z):.1f} mm")
