"""腿的真实竖向行程（第三版）：直接钉 qpos，绕开一切执行器语义。

前两版的教训：
  1. 纯正向运动学：被动闭合关节置 0，平行四边形约束不成立 → 行程只有 70 mm，假
  2. 用 d.ctrl 拉腿：执行器是**速度伺服**不是位置伺服，腿一路顶到限位
     （回读 hip=+1.513 而设定是 +0.50）→ 行程 558 mm，也是假

这一版的做法：
  · 每步把机身钉在 PIN_Z，并把 4 个腿关节的 qpos 直接写成目标角、dof 速度清零
    （等价于无穷刚度的位置伺服，与执行器类型无关）
  · 被动的 *_passive 关节不碰，让 MuJoCo 的 mjEQ_CONNECT 自己解闭合
  · 稳定后量轮子最低点 → 机身离地高度
  · 打印闭合残差 |efc_pos|，残差小才说明读数可信

用法：
  python -m diablo_training.leg_range
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from diablo_training.model import DEFAULT_Q, make_model

LEG_JOINTS = (
    ("left_hip_motor", 0), ("left_knee_motor", 1),
    ("right_hip_motor", 0), ("right_knee_motor", 1),
)
PIN_Z = 0.5
SETTLE_STEPS = 800


def _i(x) -> int:
    return int(np.asarray(x).reshape(-1)[0])


def _f(x) -> float:
    return float(np.asarray(x).reshape(-1)[0])


_G = {
    name: _i(getattr(mujoco.mjtGeom, name))
    for name in ("mjGEOM_SPHERE", "mjGEOM_CAPSULE", "mjGEOM_CYLINDER",
                 "mjGEOM_BOX", "mjGEOM_MESH", "mjGEOM_PLANE", "mjGEOM_ELLIPSOID")
}


def _geom_z_extent(m, d, i: int) -> tuple[float, float] | None:
    t = _i(m.geom_type[i])
    if t == _G["mjGEOM_PLANE"]:
        return None
    p = np.asarray(d.geom_xpos[i], dtype=float)
    rot = np.asarray(d.geom_xmat[i], dtype=float).reshape(3, 3)
    s = np.asarray(m.geom_size[i], dtype=float)
    if t == _G["mjGEOM_SPHERE"]:
        ext = float(s[0])
    elif t in (_G["mjGEOM_CYLINDER"], _G["mjGEOM_CAPSULE"]):
        axis = rot[:, 2]
        ext = abs(float(axis[2])) * float(s[1]) + float(s[0]) * math.sqrt(
            max(0.0, 1.0 - float(axis[2]) ** 2))
    elif t == _G["mjGEOM_BOX"]:
        ext = float(np.abs(rot[2, :]) @ s)
    elif t == _G["mjGEOM_ELLIPSOID"]:
        ext = float(np.sqrt(np.sum((rot[2, :] * s) ** 2)))
    elif t == _G["mjGEOM_MESH"]:
        mid = _i(m.geom_dataid[i])
        adr, num = _i(m.mesh_vertadr[mid]), _i(m.mesh_vertnum[mid])
        verts = np.asarray(m.mesh_vert[adr: adr + num], dtype=float)
        scale = s if (s.size == 3 and np.any(np.abs(s - 1.0) > 1e-9)) else np.ones(3)
        z = (verts * scale) @ rot[2, :] + p[2]
        return float(z.min()), float(z.max())
    else:
        ext = _f(m.geom_rbound[i])
    return p[2] - ext, p[2] + ext


def settle(m, d, hip: float, knee: float) -> dict[str, float]:
    """钉住机身 + 钉住 4 个腿关节，让闭合约束自己收敛。"""
    mujoco.mj_resetData(m, d)
    a = _i(m.jnt_qposadr[_i(m.joint("floating_base").id)])

    pins = []
    for jname, which in LEG_JOINTS:
        jid = _i(m.joint(jname).id)
        pins.append((_i(m.jnt_qposadr[jid]), _i(m.jnt_dofadr[jid]),
                     hip if which == 0 else knee))

    for _ in range(SETTLE_STEPS):
        d.qpos[a : a + 3] = [0.0, 0.0, PIN_Z]
        d.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[0:6] = 0.0
        for qadr, dadr, ang in pins:
            d.qpos[qadr] = ang
            d.qvel[dadr] = 0.0
        mujoco.mj_step(m, d)

    # 最后再钉一次并只做 forward：步进积分会让钉住的关节漂 1e-5，
    # 这一步保证回读和目标完全一致，同时刷新约束残差。
    d.qpos[a : a + 3] = [0.0, 0.0, PIN_Z]
    d.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[0:6] = 0.0
    for qadr, dadr, ang in pins:
        d.qpos[qadr] = ang
        d.qvel[dadr] = 0.0
    mujoco.mj_forward(m, d)

    resid = float(np.max(np.abs(np.asarray(d.efc_pos[: m.neq], dtype=float)))) \
        if m.neq else 0.0

    wb, hi = math.inf, -math.inf
    for i in range(m.ngeom):
        if _i(m.geom_bodyid[i]) == 0:
            continue
        ext = _geom_z_extent(m, d, i)
        if ext is None:
            continue
        if "wheel" in (m.body(_i(m.geom_bodyid[i])).name or "").lower():
            wb = min(wb, ext[0])
        hi = max(hi, ext[1])

    base_z = _f(d.xpos[_i(m.jnt_bodyid[_i(m.joint("floating_base").id)])][2])
    return {
        "base_over_ground_mm": (base_z - wb) * 1000.0,
        "total_height_mm": (hi - wb) * 1000.0,
        "resid_mm": resid * 1000.0,
        "q_hip": _f(d.qpos[pins[0][0]]),
        "q_knee": _f(d.qpos[pins[1][0]]),
    }


def main() -> None:
    m = make_model()
    d = mujoco.MjData(m)

    print("=" * 74)
    print("1. 执行器实况（说明前两版为什么会错）")
    print("=" * 74)
    print(f"timestep={m.opt.timestep}  nq={m.nq}  nv={m.nv}  nu={m.nu}  neq={m.neq}")
    for i in range(m.nu):
        jid = _i(m.actuator_trnid[i][0])
        print(f"  ctrl[{i}] {m.joint(jid).name:<18} "
              f"gain={np.round(np.asarray(m.actuator_gainprm[i][:2]), 2).tolist()} "
              f"bias={np.round(np.asarray(m.actuator_biasprm[i][:3]), 2).tolist()} "
              f"gear={np.round(np.asarray(m.actuator_gear[i][:2]), 3).tolist()}")
    print("  提示：bias 第 3 项为 -1 → 速度伺服；为 -kp/-kd 形式 → 位置伺服")

    print("\n2. 校验：钉住的关节能不能保持住目标角")
    for hip, knee in ((0.5, -1.0), (0.0, 0.0), (0.9, -2.0)):
        r = settle(m, d, hip, knee)
        ok = abs(r["q_hip"] - hip) < 1e-6 and abs(r["q_knee"] - knee) < 1e-6
        print(f"  设定 hip={hip:+.2f} knee={knee:+.2f} → "
              f"回读 {r['q_hip']:+.3f} / {r['q_knee']:+.3f}  "
              f"{'一致' if ok else '不一致！'}   "
              f"机身离地 {r['base_over_ground_mm']:.1f} mm  残差 {r['resid_mm']:.2f} mm")

    print("\n3. hip × knee 网格：机身离地高度 mm（覆盖关节全行程）")
    knees = np.linspace(-2.85, 0.05, 8)
    hips = np.linspace(-1.50, 1.50, 8)
    print("   hip\\knee " + "".join(f"{k:>8.2f}" for k in knees))
    vals = []
    for hip in hips:
        cells = []
        for knee in knees:
            r = settle(m, d, float(hip), float(knee))
            cells.append(f"{r['base_over_ground_mm']:>8.1f}")
            if r["resid_mm"] < 5.0:
                vals.append((r["base_over_ground_mm"], r["total_height_mm"],
                             float(hip), float(knee), r["resid_mm"]))
        print(f"   {hip:>+8.2f} " + "".join(cells))

    print("\n4. 行程")
    if vals:
        lo = min(vals, key=lambda v: v[0])
        hi = max(vals, key=lambda v: v[0])
        print(f"  可信样本 {len(vals)} 个（闭合残差 < 5 mm）")
        print(f"  最低：机身离地 {lo[0]:.1f} mm  整机高 {lo[1]:.1f} mm  "
              f"(hip={lo[2]:+.2f}, knee={lo[3]:+.2f})")
        print(f"  最高：机身离地 {hi[0]:.1f} mm  整机高 {hi[1]:.1f} mm  "
              f"(hip={hi[2]:+.2f}, knee={hi[3]:+.2f})")
        print(f"  竖向行程 = {hi[0] - lo[0]:.1f} mm")
    else:
        print("  没有配置的闭合残差 < 5 mm，读数不可信")

    r = settle(m, d, DEFAULT_Q[0], DEFAULT_Q[1])
    print(f"\n5. DEFAULT_Q({DEFAULT_Q[0]:+.2f}, {DEFAULT_Q[1]:+.2f})："
          f"机身离地 {r['base_over_ground_mm']:.1f} mm  "
          f"整机高 {r['total_height_mm']:.1f} mm  残差 {r['resid_mm']:.2f} mm")
    print("  对照：规格书 站立 491 / 匍匐 270 → 行程 221 mm")
    print("  对照：训练配置 min_height=0.16 max_height=0.34 → 指令 180 mm")


if __name__ == "__main__":
    main()
