"""Independent published-transform, virtual-work and actual MuJoCo drive checks."""

import re

import mujoco
import numpy as np

from .control import decode_action, radial_feedforward
from .model import DEFAULT_Q, JOINTS, closure_errors, make_model, metadata, set_pose
from .resources import mechanism_path


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def published_frames():
    # Read the independent Webots source, not the generated MuJoCo model.
    text = mechanism_path().read_text()
    frames = []
    for side in ("left", "right"):
        marker = text.index(f'name "{side}_j4"')
        hinge = text.rfind("jointParameters HingeJointParameters", 0, marker)
        block = text[hinge:marker]
        axis = np.fromstring(re.search(r"axis ([^\n]+)", block)[1], sep=" ")
        anchor = np.fromstring(re.search(r"anchor ([^\n]+)", block)[1], sep=" ")
        endpoint = text[text.index("endPoint Solid", marker) :]
        aa = np.fromstring(re.search(r"rotation ([^\n]+)", endpoint)[1], sep=" ")
        frames.append((anchor, axis, rotation(aa[:3], aa[3])))
    # Assert the independently published closure anchors and serial-link offsets.
    for declaration in (
        "translation 0.140000 0.000000 0.000000",
        "translation 0.090000 0.000000 0.000000",
        "rotation 0.000000 0.000000 -1.000000 2.872900",
        "translation 0.140000 0.000000 0.053700",
        "translation 0.140000 0.000000 -0.053700",
    ):
        if declaration not in text:
            raise AssertionError(
                f"Published mechanism reference changed: {declaration}"
            )
    return frames


def validate_model():
    model = make_model()
    data = mujoco.MjData(model)
    frames = published_frames()
    ids = [model.joint(n).id for n in JOINTS]
    vi = model.jnt_dofadr[ids]
    assert model.nu == 6 and model.neq == 2 and model.nv == 16
    assert np.all(model.body_mass[1:] > 0) and np.all(model.body_inertia[1:] > 0)
    np.testing.assert_allclose(
        model.body_ipos[model.body("diablo_base_link").id],
        [-0.001, 0.0, 0.009267],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        model.body_ipos[model.body("left_fake_thigh_link").id],
        [0.001204, 0.0, -0.066749],
        atol=1e-8,
    )
    np.testing.assert_allclose(
        model.body_ipos[model.body("left_shank_link").id],
        [0.092381, 0.000001, 0.012349],
        atol=1e-8,
    )
    max_fk = max_loop = max_virtual_work = 0.0
    samples = 0
    # Wide non-singular configurations, both sides, not only the reset pose.
    for hip in (-0.6, 0.0, 0.5, 1.0):
        for knee in (-2.0, -1.0, -0.3):
            q = np.array([hip, knee, 0.3, hip, knee, -0.2])
            mujoco.mj_resetData(model, data)
            data.qpos[:3] = [0, 0, 1]
            set_pose(model, data, q)
            mujoco.mj_forward(model, data)
            max_loop = max(max_loop, *closure_errors(model, data))
            for side, (anchor, axis, r0), offset in zip(
                ("left", "right"), frames, (0.0537, -0.0537)
            ):
                motor_rotation = rotation(axis, hip) @ r0
                expected = (
                    anchor
                    + motor_rotation
                    @ (
                        rotation([0, 0, 1], -2.8729 - knee) @ [0.14, 0, 0]
                        + np.array([0.14, 0, offset])
                    )
                    + [0, 0, 1]
                )
                actual = data.body(f"{side}_wheel").xpos
                max_fk = max(max_fk, float(np.linalg.norm(actual - expected)))

            # Leg length from actual model endpoints, finite difference vs F*dl/dq.
            def length(k):
                shifted = q.copy()
                shifted[[1, 4]] = k
                set_pose(model, data, shifted)
                mujoco.mj_forward(model, data)
                delta = (
                    data.body("left_wheel").xpos
                    - data.body("left_fake_thigh_link").xpos
                )
                return np.sqrt(np.dot(delta, delta) - 0.0537**2)

            derivative = (length(knee + 1e-6) - length(knee - 1e-6)) / 2e-6
            max_virtual_work = max(
                max_virtual_work, abs(60 * derivative - radial_feedforward(q)[1])
            )
            samples += 1
    assert max_fk < 3e-6, max_fk
    assert max_loop < 1e-9, max_loop
    assert max_virtual_work < 1e-6, max_virtual_work
    # Actuation power and direction in the real compiled mass/constraint system.
    model.opt.gravity[:] = 0
    impulse_accelerations = []
    for i in range(6):
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 1
        set_pose(model, data, DEFAULT_Q)
        data.ctrl[i] = 1.0
        mujoco.mj_forward(model, data)
        impulse_accelerations.append(float(data.qacc[vi[i]]))
        assert data.qacc[vi[i]] > 0, (JOINTS[i], data.qacc[vi[i]])
        np.testing.assert_allclose(data.qfrc_actuator[vi], np.eye(6)[i], atol=1e-12)
        data.qvel[vi] = np.arange(6) * 0.2
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            data.qfrc_actuator @ data.qvel,
            data.actuator_force @ data.actuator_velocity,
            atol=1e-12,
        )
    # All six motor axes point approximately -body-y. Wheel policy positive is raw negative.
    set_pose(model, data, DEFAULT_Q)
    mujoco.mj_forward(model, data)
    for jid in ids:
        np.testing.assert_allclose(data.xaxis[jid], [0, -1, 0], atol=2e-5)
    _, applied = decode_action(np.ones(6), DEFAULT_Q, np.zeros(6), "torque")
    rolling_direction = []
    for i in (2, 5):
        omega = data.xaxis[ids[i]] * np.sign(applied[i])
        # No-slip center velocity = -omega cross bottom contact vector.
        rolling_direction.append(float(-np.cross(omega, [0, 0, -0.1])[0]))
    assert min(rolling_direction) > 0.099
    return {
        "checks": "passed",
        "configurations_checked": samples,
        "fk_max_error_m": max_fk,
        "initial_closure_max_error_m": max_loop,
        "virtual_work_max_error_nm": max_virtual_work,
        "positive_motor_unit_torque_accel_rad_s2": impulse_accelerations,
        "positive_wheel_policy_rolling_direction_x": rolling_direction,
        "model": metadata(model),
        "scope": "Published reference simulation mechanics; not hardware calibration or learned locomotion",
    }
