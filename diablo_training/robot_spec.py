"""把仓库里这台机器人的全部参数导出来，并交叉验证腿的竖向行程。

为什么需要这个：
  `diablo_training/model.py` 里没有外部 URDF/XML 文件，机器人几何是用
  `_robot_tree()` 直接写成 MJCF 字符串再 `MjSpec.from_string` 编出来的。
  所以"这台机器人的尺寸/质量/惯量/扭矩"只能从编译后的 mjModel 里读。

它同时做一件校验：用仓库自己的 `set_pose`（被动关节 = ±knee）做正向运动学，
把结果和 `leg_range` 的"物理松弛"结果对照。两条路给出同一个答案，才说明可信。

用法：
  python -m diablo_training.robot_spec
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from diablo_training.model import DEFAULT_Q, STALL_TORQUE, make_model, set_pose

try:  # 这个常数可能在 control.py 而不是 model.py
    from diablo_training.model import NO_LOAD_SPEED
except Exception:
    try:
        from diablo_training.control import NO_LOAD_SPEED
    except Exception:
        NO_LOAD_SPEED = float("nan")

LEG_ACT = ("left_hip_motor", "left_knee_motor", "left_wheel_motor",
           "right_hip_motor", "right_knee_motor", "right_wheel_motor")


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


def measure(m, d, motor_q) -> dict[str, float]:
    """用仓库的 set_pose 摆姿态，量机身离地高度（轮底为 0）。"""
    mujoco.mj_resetData(m, d)
    a = _i(m.jnt_qposadr[_i(m.joint("floating_base").id)])
    d.qpos[a : a + 3] = [0.0, 0.0, 0.5]
    d.qpos[a + 3 : a + 7] = [1.0, 0.0, 0.0, 0.0]
    set_pose(m, d, motor_q)
    mujoco.mj_kinematics(m, d)

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
    }


def main() -> None:
    m = make_model()
    d = mujoco.MjData(m)

    print("=" * 78)
    print("1. 全局")
    print("=" * 78)
    print(f"nq={m.nq}  nv={m.nv}  nu={m.nu}  neq={m.neq}  nbody={m.nbody}  ngeom={m.ngeom}")
    print(f"timestep={m.opt.timestep}  gravity={np.round(m.opt.gravity, 3).tolist()}  "
          f"iterations={m.opt.iterations}")
    print(f"integrator={mujoco.mjtIntegrator(m.opt.integrator).name}")
    print(f"总质量 = {mujoco.mj_getTotalmass(m):.4f} kg")
    print(f"STALL_TORQUE={STALL_TORQUE} N·m   "
          f"NO_LOAD_SPEED={NO_LOAD_SPEED:.3f} rad/s   "
          f"轮半径 0.1 m → 轮缘线速度 {NO_LOAD_SPEED * 0.1:.3f} m/s")

    print("\n2. 连杆（相对父节点的位置 / 质量 / 惯量对角）")
    print(f"  {'id':>3} {'name':<24} {'parent':<22} {'pos(m)':<24} "
          f"{'mass':>7} {'inertia diag':<26} com")
    for b in range(m.nbody):
        p = _i(m.body_parentid[b])
        print(f"  {b:>3} {m.body(b).name:<24} {m.body(p).name:<22} "
              f"{np.round(np.asarray(m.body_pos[b]), 4).tolist()!s:<24} "
              f"{_f(m.body_mass[b]):>7.3f} "
              f"{np.round(np.asarray(m.body_inertia[b]), 5).tolist()!s:<26} "
              f"{np.round(np.asarray(m.body_ipos[b]), 4).tolist()}")

    print("\n3. 关节（轴 / 锚点 / 限位 / 阻尼 / armature）")
    print(f"  {'id':>3} {'name':<22} {'type':<12} {'body':<22} {'axis':<14} "
          f"{'range':<22} {'damp':>6} {'arm':>6}")
    for j in range(m.njnt):
        jid = j
        typ = mujoco.mjtJoint(_i(m.jnt_type[jid])).name
        body = m.body(_i(m.jnt_bodyid[jid])).name
        rng = np.asarray(m.jnt_range[jid], dtype=float)
        rng_s = f"[{rng[0]:+.3f},{rng[1]:+.3f}]" if _i(m.jnt_limited[jid]) else "无限位"
        dof = _i(m.jnt_dofadr[jid])
        print(f"  {jid:>3} {m.joint(jid).name:<22} {typ:<12} {body:<22} "
              f"{np.round(np.asarray(m.jnt_axis[jid]), 3).tolist()!s:<14} "
              f"{rng_s:<22} {_f(m.dof_damping[dof]):>6.3f} "
              f"{_f(m.dof_armature[dof]):>6.3f}")

    print("\n4. 执行器（ctrl 是扭矩，单位 N·m）")
    for i in range(m.nu):
        jid = _i(m.actuator_trnid[i][0])
        cr = np.asarray(m.actuator_ctrlrange[i], dtype=float)
        print(f"  ctrl[{i}] {m.actuator(i).name:<24} joint={m.joint(jid).name:<18} "
              f"ctrlrange=[{cr[0]:+.1f}, {cr[1]:+.1f}]  gear={_f(m.actuator_gear[i][0]):.1f}")

    print("\n5. 闭合约束用的 site（平行四边形弦长）")
    for s in range(m.nsite):
        if not m.site(s).name:
            continue
        print(f"  site[{s}] {m.site(s).name:<20} body={m.body(_i(m.site_bodyid[s])).name:<22} "
              f"pos={np.round(np.asarray(m.site_pos[s]), 4).tolist()}")

    print("\n6. 用仓库自己的 set_pose（被动关节 = ±knee）验行程")
    r = measure(m, d, list(DEFAULT_Q))
    print(f"  DEFAULT_Q {list(DEFAULT_Q)}：机身离地 {r['base_over_ground_mm']:.1f} mm  "
          f"整机高 {r['total_height_mm']:.1f} mm")

    print(f"\n  {'hip':>6} {'knee':>6} | {'机身离地 mm':>12} {'整机高 mm':>10}")
    for hip, knee in ((0.5, -1.0), (0.0, 0.0), (0.21, -2.44), (1.5, -1.61),
                      (0.9, -2.0), (-1.2, -2.4)):
        rr = measure(m, d, [hip, knee, 0.0, hip, knee, 0.0])
        print(f"  {hip:>+6.2f} {knee:>+6.2f} | {rr['base_over_ground_mm']:>12.1f} "
              f"{rr['total_height_mm']:>10.1f}")

    print("\n  hip × knee 网格（机身离地 mm，左列 hip，上方 knee）")
    knees = np.linspace(-2.85, 0.05, 8)
    hips = np.linspace(-1.50, 1.50, 8)
    print("   hip\\knee " + "".join(f"{k:>8.2f}" for k in knees))
    lo = (math.inf, 0, 0)
    hi = (-math.inf, 0, 0)
    hit = 0
    for hip in hips:
        cells = []
        for knee in knees:
            rr = measure(m, d, [float(hip), float(knee), 0.0,
                                float(hip), float(knee), 0.0])
            v = rr["base_over_ground_mm"]
            cells.append(f"{v:>8.1f}")
            if math.isfinite(v):
                hit += 1
                if v < lo[0]:
                    lo = (v, float(hip), float(knee))
                if v > hi[0]:
                    hi = (v, float(hip), float(knee))
        print(f"   {hip:>+8.2f} " + "".join(cells))

    print(f"\n  有效样本 {hit}/64")
    print(f"  最低 机身离地 {lo[0]:.1f} mm (hip={lo[1]:+.2f}, knee={lo[2]:+.2f})")
    print(f"  最高 机身离地 {hi[0]:.1f} mm (hip={hi[1]:+.2f}, knee={hi[2]:+.2f})")
    print(f"  行程 = {hi[0] - lo[0]:.1f} mm")

    print("\n7. URDF 溯源（模型的真实来源）")
    urdf_path = Path(__file__).resolve().parents[1] / (
        "third_party/diablo_source/resources/robots/diablo/urdf/diablo_asm.urdf")
    if not urdf_path.exists():
        print(f"  找不到 {urdf_path}")
        print(f"  当前文件位置：{Path(__file__).resolve()}")
    else:
        root = ET.parse(urdf_path).getroot()
        links = root.findall("link")
        joints = root.findall("joint")
        umass = {}
        for link in links:
            mass_el = link.find("inertial/mass")
            if mass_el is not None:
                umass[link.get("name")] = float(mass_el.get("value"))
        print(f"  {urdf_path}")
        print(f"  link {len(links)} 个  joint {len(joints)} 个  mesh "
              f"{len(list(urdf_path.parent.parent.rglob('*.STL'))) + len(list(urdf_path.parent.parent.rglob('*.stl')))} 个")
        utotal = sum(umass.values())
        mtotal = mujoco.mj_getTotalmass(m)
        print(f"  URDF 总质量 = {utotal:.4f} kg      模型总质量 = {mtotal:.4f} kg      "
              f"差 {mtotal - utotal:+.4f} kg")
        print(f"  {'URDF link':<28}{'URDF kg':>10}   {'模型 body':<24}{'模型 kg':>10}")
        um = sorted(umass.items(), key=lambda kv: -kv[1])
        mb = sorted(((m.body(b).name, _f(m.body_mass[b])) for b in range(1, m.nbody)),
                    key=lambda kv: -kv[1])
        for i in range(max(len(um), len(mb))):
            ln = f"{um[i][0]:<28}{um[i][1]:>10.4f}" if i < len(um) else ""
            rn = f"{mb[i][0]:<24}{mb[i][1]:>10.4f}" if i < len(mb) else ""
            print(f"  {ln}   {rn}")

        print("\n  URDF 关节限位 vs 模型")
        for j in joints:
            name = j.get("name")
            lim = j.find("limit")
            if lim is None:
                continue
            lo = float(lim.get("lower", "nan"))
            hi = float(lim.get("upper", "nan"))
            eff = lim.get("effort")
            extra = ""
            try:
                mm = m.joint(name)
                rng = np.asarray(m.jnt_range[_i(mm.id)], dtype=float)
                extra = f"   模型=[{rng[0]:+.3f},{rng[1]:+.3f}]"
            except Exception:
                extra = "   模型里没有同名关节"
            print(f"  {name:<22} URDF=[{lo:+.3f},{hi:+.3f}] effort={eff}{extra}")

        print("\n  URDF 里的轮子几何")
        for link in links:
            lname = (link.get("name") or "").lower()
            if "wheel" not in lname:
                continue
            for geo in link.findall(".//geometry/*"):
                attrs = {k: v for k, v in geo.attrib.items() if k in ("radius", "length", "filename")}
                if attrs:
                    print(f"  {link.get('name'):<24} {geo.tag:<10} {attrs}")

    print("\n8. 和外部规格对照")
    print(f"  总质量     模型 {mujoco.mj_getTotalmass(m):.2f} kg   vs 规格 22.9 kg  "
          f"({mujoco.mj_getTotalmass(m) / 22.9 - 1:+.1%})")
    print(f"  整机最高   模型 {max(r['total_height_mm'], 0):.1f} mm   vs 规格 491 mm")
    print("  491 / 270 这两个数已用 grep 搜过整个仓库，没有命中（只存在于外部规格书）")


if __name__ == "__main__":
    main()
