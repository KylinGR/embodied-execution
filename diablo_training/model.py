"""Translate the upstream equivalent URDF using MuJoCo's native importer."""

import hashlib
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .resources import verify_source

JOINTS = (
    "left_fake_hip_joint",
    "left_fake_knee_joint",
    "left_wheel_joint",
    "right_fake_hip_joint",
    "right_fake_knee_joint",
    "right_wheel_joint",
)
WHEELS = np.array([2, 5])
LEGS = np.array([0, 1, 3, 4])
DEFAULT_Q = np.array([-0.5, 1.0, 0.0, -0.5, 1.0, 0.0])
PHYSICS_DT = 0.002
CONTROL_DT = 0.01


def make_model():
    source = verify_source()
    robot = ET.parse(source).getroot()
    # Resolve original filenames, including their case, without copying meshes.
    for mesh in robot.findall(".//mesh"):
        mesh.set("filename", str((source.parent / mesh.get("filename")).resolve()))
    extension = ET.SubElement(robot, "mujoco")
    ET.SubElement(extension, "compiler", fusestatic="false", discardvisual="false")
    spec = mujoco.MjSpec.from_string(ET.tostring(robot, encoding="unicode"))
    base = spec.body("diablo_base_link")
    base.add_freejoint(name="floating_base")
    base.pos = [0, 0, 0.4]
    spec.option.timestep = PHYSICS_DT
    spec.option.gravity = [0, 0, -9.81]
    spec.option.iterations = 50
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[20, 20, 0.1],
        rgba=[0.23, 0.26, 0.30, 1],
        contype=2,
        conaffinity=1,
        friction=[0.8, 0.005, 0.0001],
    )
    spec.worldbody.add_light(pos=[0, -3, 4], dir=[0, 0, -1])
    site = base.add_site(name="body_imu", size=[0.005, 0.005, 0.005])
    for name, kind in (
        ("gyro", mujoco.mjtSensor.mjSENS_GYRO),
        ("accelerometer", mujoco.mjtSensor.mjSENS_ACCELEROMETER),
    ):
        spec.add_sensor(
            name=name, type=kind, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=site.name
        )
    for name in JOINTS:
        joint = spec.joint(name)
        joint.damping = [0.05, 0, 0]
        joint.armature = 0.01
        if "wheel" in name:
            joint.limited = False
        spec.add_actuator(
            name=name + "_motor",
            trntype=mujoco.mjtTrn.mjTRN_JOINT,
            target=name,
            gear=[1, 0, 0, 0, 0, 0],
            ctrllimited=True,
            ctrlrange=[-27, 27],
            forcelimited=True,
            forcerange=[-27, 27],
        )
    for geom in spec.geoms:
        if geom.name != "floor" and (geom.contype or geom.conaffinity):
            # Match the training source's disabled robot self-collisions.
            geom.contype = 1
            geom.conaffinity = 2
            geom.friction = [0.8, 0.005, 0.0001]
    model = spec.compile()
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    return model


def metadata(model):
    source = verify_source()
    return {
        "source": "DDTRobot/Col_DIABLO_Issac_RL/resources/robots/diablo/urdf/diablo_asm.urdf",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "model_scope": "upstream serial equivalent; no parallel linkage equality constraints",
        "joints": list(JOINTS),
        "actuators": model.nu,
        "default_joint_rad": DEFAULT_Q.tolist(),
        "mass_kg": float(model.body_mass.sum()),
        "physics_dt_s": PHYSICS_DT,
        "control_dt_s": CONTROL_DT,
        "simulation_only": True,
        "hardware_authorized": False,
        "limits_scope": "27 Nm and 26 rad/s from upstream URDF, not hardware certification",
    }
