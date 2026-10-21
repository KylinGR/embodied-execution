"""CAD-based DIABLO with serial motor drives and parallelogram closures."""

from copy import deepcopy
import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .resources import mechanism_path, source_path, verify_source

JOINTS = tuple(
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("hip_motor", "knee_motor", "wheel_motor")
)
WHEELS = np.array([2, 5])
LEGS = np.array([0, 1, 3, 4])
DEFAULT_Q = np.array([0.5, -1.0, 0.0, 0.5, -1.0, 0.0])
OBSERVATION_SIGNS = np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0])
PHYSICS_DT = 0.002
CONTROL_DT = 0.01
STALL_TORQUE = 17.0
NO_LOAD_SPEED = 220 * 2 * np.pi / 60


def _robot_tree():
    source = verify_source()
    robot = ET.parse(source).getroot()
    equivalent = ET.parse(source.with_name("diablo_asm.urdf")).getroot()
    for side in ("left", "right"):
        for old, new, kind in (
            (f"{side}_fake_hip_joint", f"{side}_hip_motor", "revolute"),
            (f"{side}_hip_joint", f"{side}_knee_motor", "revolute"),
            (f"{side}_knee_joint", f"{side}_return_passive", "continuous"),
            (f"{side}_wheel_joint", f"{side}_wheel_motor", "continuous"),
        ):
            joint = robot.find(f"joint[@name='{old}']")
            joint.set("name", new)
            joint.set("type", kind)
            axis = joint.find("axis")
            if axis is None:
                axis = ET.SubElement(joint, "axis")
            axis.set("xyz", "0 0 1" if "passive" in new else "0 0 -1")
            limit = joint.find("limit")
            if limit is not None:
                joint.remove(limit)
            if kind == "revolute":
                limits = (-1.57, 1.57) if "hip" in new else (-2.9, 0.1)
                ET.SubElement(
                    joint,
                    "limit",
                    lower=str(limits[0]),
                    upper=str(limits[1]),
                    effort=str(STALL_TORQUE),
                    velocity=str(NO_LOAD_SPEED),
                )
        branch = deepcopy(equivalent.find(f"link[@name='{side}_fake_shank_link']"))
        branch.set("name", f"{side}_parallel_link")
        branch.find("inertial/origin").set(
            "xyz", f"0.072113 0.016825 {'-0.008' if side == 'left' else '0.008'}"
        )
        robot.append(branch)
        joint = ET.SubElement(
            robot, "joint", name=f"{side}_branch_passive", type="continuous"
        )
        ET.SubElement(joint, "parent", link=f"{side}_fake_thigh_link")
        ET.SubElement(joint, "child", link=f"{side}_parallel_link")
        ET.SubElement(joint, "origin", xyz="0.09 0 0", rpy="0 0 -2.8729")
        ET.SubElement(joint, "axis", xyz="0 0 1")
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="false")
    return robot


def _robot_spec():
    """Build the validated robot and common sensors/constraints, without terrain."""
    spec = mujoco.MjSpec.from_string(ET.tostring(_robot_tree(), encoding="unicode"))
    base = spec.body("diablo_base_link")
    base.add_freejoint(name="floating_base")
    base.pos = [0, 0, 0.4]
    spec.option.timestep = PHYSICS_DT
    spec.option.gravity = [0, 0, -9.81]
    spec.option.iterations = 80
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_AUTORESET
    spec.worldbody.add_light(pos=[0, -3, 4], dir=[0, 0, -1])
    site = base.add_site(name="body_imu", size=[0.005] * 3)
    for name, kind in (
        ("gyro", mujoco.mjtSensor.mjSENS_GYRO),
        ("accelerometer", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
    ):
        spec.add_sensor(
            name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name
        )
    for side in ("left", "right"):
        spec.body(f"{side}_parallel_link").add_site(
            name=f"{side}_loop_a", pos=[0.14, 0, 0], size=[0.002] * 3
        )
        spec.body(f"{side}_shank_link").add_site(
            name=f"{side}_loop_b", pos=[0.09, 0, 0], size=[0.002] * 3
        )
        spec.add_equality(
            name=f"{side}_parallelogram",
            type=mujoco.mjtEq.mjEQ_CONNECT,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            name1=f"{side}_loop_a",
            name2=f"{side}_loop_b",
            solref=[0.004, 1],
            solimp=[0.99, 0.999, 0.001, 0.5, 2],
        )
    for joint in spec.joints:
        if "passive" in joint.name:
            joint.damping = [0.001, 0, 0]
    for name in JOINTS:
        joint = spec.joint(name)
        joint.damping = [0.05, 0, 0]
        joint.armature = 0.01
        spec.add_actuator(
            name=name + "_actuator",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=name,
            gear=[1, 0, 0, 0, 0, 0],
            ctrllimited=True,
            ctrlrange=[-STALL_TORQUE, STALL_TORQUE],
            forcelimited=True,
            forcerange=[-STALL_TORQUE, STALL_TORQUE],
        )
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 1, 2
            geom.friction = [0.8, 0.005, 0.0001]
    return spec


def _finish_model(spec):
    model = spec.compile()
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 720
    return model


def make_model():
    spec = _robot_spec()
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20, 20, 0.1],
        rgba=[0.23, 0.26, 0.30, 1],
        contype=2,
        conaffinity=1,
        friction=[0.8, 0.005, 0.0001],
    )
    return _finish_model(spec)


def set_pose(model, data, motor_q):
    """Initialize dependent joints on the published parallelogram branch."""
    for i, side in enumerate(("left", "right")):
        h, k, w = motor_q[3 * i : 3 * i + 3]
        for name, value in (
            ("hip_motor", h),
            ("knee_motor", k),
            ("wheel_motor", w),
            ("return_passive", k),
            ("branch_passive", -k),
        ):
            data.qpos[model.jnt_qposadr[model.joint(f"{side}_{name}").id]] = value


def closure_errors(model, data):
    return [
        float(
            np.linalg.norm(
                data.site(f"{s}_loop_a").xpos - data.site(f"{s}_loop_b").xpos
            )
        )
        for s in ("left", "right")
    ]


def metadata(model):
    source = verify_source()
    _, lock = source_path()
    return {
        "source": lock["robot"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "upstream_commit": lock["commit"],
        "mechanism_commit": lock["mechanism"]["commit"],
        "mechanism_sha256": hashlib.sha256(mechanism_path().read_bytes()).hexdigest(),
        "model_scope": "CAD reference with two explicit parallelogram closures; not hardware calibrated",
        "joints": list(JOINTS),
        "actuators": model.nu,
        "closure_constraints": model.neq,
        "default_joint_rad": DEFAULT_Q.tolist(),
        "observation_velocity_signs": OBSERVATION_SIGNS.tolist(),
        "mass_kg": float(model.body_mass.sum()),
        "physics_dt_s": PHYSICS_DT,
        "control_dt_s": CONTROL_DT,
        "simulation_only": True,
        "hardware_authorized": False,
        "motor_reference": {
            "source": "arXiv:2407.21500 Table II",
            "stall_nm": STALL_TORQUE,
            "no_load_rad_s": NO_LOAD_SPEED,
            "rated_nm": 9.6,
        },
        "limits_scope": "Published reference motor, assumed linear torque-speed envelope; not firmware/thermal calibration",
    }
