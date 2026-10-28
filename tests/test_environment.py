"""Dependency-free course tests plus CPU MuJoCo regression tests."""

import ast
from pathlib import Path
import tomllib
import unittest

from diablo_training.contract import (
    BLIND_SEEDS,
    SCREEN_SEEDS,
    VERIFY_SEEDS,
    gate_failures,
    gate_score,
    interface_contract,
    load_course,
    metric_name,
    profile_command,
    select_policy_best,
    verify_interface_contract,
)

try:
    import mujoco
    import numpy as np

    from diablo_training.control import decode_action
    from diablo_training.env import Config, DiabloEnv, command_profile
    from diablo_training.model import (
        DEFAULT_Q,
        JOINTS,
        NO_LOAD_SPEED,
        OFFICIAL_MASS_KG,
        STALL_TORQUE,
        TARGET_BASE_HEIGHT_M,
    )
    from diablo_training.validation import validate_model

    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


COMPOSITE_METRICS = {
    "pass_rate": 1.0,
    "duration_fraction": 1.0,
    "speed_mae_mps": 0.1,
    "yaw_mae_rad_s": 0.08,
    "upright": 0.99,
    "mean_height_error_m": 0.01,
    "height_deviation_m": 0.02,
    "wheel_slip_mps": 0.1,
    "backslide_mps": 0.05,
    "wheel_center_offset_m": 0.005,
    "straight_wheel_speed_difference_mps": 0.02,
    "forward_progress_m": 47.5,
    "obstacle_passes": 3,
    "finished": True,
    "mean_positive_power_w": 80.0,
}


class ContractTests(unittest.TestCase):
    def test_course_order_and_seed_sets(self):
        course = load_course()
        self.assertEqual(len(course.order), 18)
        self.assertEqual(course.order[0], "flat_balance")
        self.assertEqual(course.order[-1], "composite")
        self.assertEqual(SCREEN_SEEDS, (2001, 2002, 2003))
        self.assertEqual(VERIFY_SEEDS, (2004, 2005, 2006, 2007, 2008))
        self.assertFalse(set(SCREEN_SEEDS) & set(VERIFY_SEEDS))
        self.assertFalse(set(SCREEN_SEEDS + VERIFY_SEEDS) & set(BLIND_SEEDS))
        contract = interface_contract(course)
        verify_interface_contract(contract)
        tampered = dict(contract)
        tampered["action_shape"] = 7
        with self.assertRaises(ValueError):
            verify_interface_contract(tampered)
        self.assertEqual(contract["observation_shape"], 160)
        self.assertEqual(contract["action_shape"], 6)
        self.assertEqual(contract["target_base_height_m"], 0.270943)

    def test_every_gate_has_an_evaluable_metric_name(self):
        course = load_course()
        for name in course.order:
            for gate_name in course.stage(name)["gate"]:
                self.assertTrue(metric_name(gate_name))
        reward_source = ast.parse(
            (Path(__file__).resolve().parents[1] / "diablo_training" / "mjlab_tasks.py")
            .read_text(encoding="utf-8")
        )
        metric_keys: set[str] = set()
        for node in ast.walk(reward_source):
            if isinstance(node, ast.FunctionDef) and node.name == "metrics":
                for child in ast.walk(node):
                    if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                        metric_keys = {
                            key.value
                            for key in child.value.keys
                            if isinstance(key, ast.Constant)
                        }
        required = {
            metric_name(gate_name)
            for name in course.order
            for gate_name in course.stage(name)["gate"]
        }
        self.assertEqual(required - metric_keys, set())

    def test_gate_scoring_and_best_selection(self):
        gate = load_course().stage("composite")["gate"]
        self.assertEqual(gate_failures(COMPOSITE_METRICS, gate), [])
        score = gate_score(COMPOSITE_METRICS, gate)
        self.assertTrue(score["passed"])
        self.assertGreater(score["score"], 0.0)
        candidates = [
            {
                "path": "a.pt",
                "iteration": 10,
                "passed": True,
                "score": 80.0,
                "worst_seed_score": 70.0,
                "minimum_margin": 0.2,
                "mean_speed_mae_mps": 0.1,
                "mean_yaw_mae_rad_s": 0.1,
            },
            {
                "path": "b.pt",
                "iteration": 20,
                "passed": True,
                "score": 90.0,
                "worst_seed_score": 60.0,
                "minimum_margin": 0.1,
                "mean_speed_mae_mps": 0.2,
                "mean_yaw_mae_rad_s": 0.2,
            },
        ]
        self.assertEqual(select_policy_best(candidates)["path"], "b.pt")
        with self.assertRaises(ValueError):
            select_policy_best([{**candidates[0], "passed": False}])

    def test_course_profile_interpolation(self):
        course = load_course()
        forward = course.stage("flat_motion_020")["profiles"][0]
        self.assertEqual(profile_command(forward, 2.0), (0.0, 0.0))
        self.assertEqual(profile_command(forward, 6.0), (0.2, 0.0))
        self.assertEqual(profile_command(forward, 99.0), (0.0, 0.0))

    def test_terrain_requirements(self):
        course = load_course()
        self.assertEqual(course.stage("slope")["terrain"]["grade_deg"], 35.0)
        self.assertEqual(course.stage("rough")["terrain"]["max_grade_deg"], 35.0)
        obstacle = course.stage("roller_obstacle")["terrain"]
        self.assertEqual(obstacle["min_height_m"], 0.02)
        self.assertEqual(obstacle["max_height_m"], 0.05)
        self.assertEqual(set(obstacle["types"]), {"box", "cylinder", "capsule"})
        composite = course.stage("composite")["terrain"]
        self.assertEqual(composite["max_grade_deg"], 35.0)
        self.assertEqual(composite["length_m"], 47.5)

    def test_gpu_training_dependency_contract(self):
        root = Path(__file__).resolve().parents[1]
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = " ".join(project["project"]["dependencies"]).lower()
        self.assertNotIn("baselines", dependencies)
        self.assertIn("mjlab==1.6.0", dependencies)


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE, "CPU MuJoCo/Gymnasium dependencies are not installed"
)
class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DiabloEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_sensor_actuator_and_height_contract(self):
        env = self.env
        observation, _ = env.reset(seed=9)
        self.assertEqual(env.model.nu, 6)
        self.assertEqual(observation.shape, (125,))
        self.assertTrue(env.observation_space.contains(observation))
        names = [env.model.joint(int(j)).name for j in env.model.actuator_trnid[:, 0]]
        self.assertEqual(names, list(JOINTS))
        self.assertEqual(env.data.ncon, 0, "Initial pose must not penetrate the ground")
        self.assertAlmostEqual(np.linalg.norm(observation[-22:-19]), 1.0, places=5)
        self.assertAlmostEqual(TARGET_BASE_HEIGHT_M, 0.270943)
        self.assertAlmostEqual(
            env.data.xpos[env.base_id, 2], TARGET_BASE_HEIGHT_M, places=4
        )
        self.assertAlmostEqual(float(env.model.body_mass.sum()), OFFICIAL_MASS_KG)

    def test_actual_physics_step_and_reset_reproducibility(self):
        env = self.env
        before, _ = env.reset(seed=10)
        action = np.array([0.1, -0.1, 0.2, 0.1, -0.1, 0.2])
        after, _, _, _, info = env.step(action)
        self.assertAlmostEqual(env.data.time, 0.01)
        self.assertTrue(np.isfinite(after).all())
        self.assertLessEqual(info["max_torque_nm"], STALL_TORQUE)
        np.testing.assert_allclose(after[-6:], action, atol=1e-7)
        again, _ = env.reset(seed=10)
        np.testing.assert_array_equal(before, again)

    def test_invalid_action_cannot_advance_physics(self):
        self.env.reset(seed=1)
        with self.assertRaises(ValueError):
            self.env.step(np.full(6, np.nan))
        self.assertEqual(self.env.data.time, 0.0)

    def test_cpu_command_profiles_and_config_validation(self):
        self.assertEqual(command_profile(5, 0, 0.5), 0.0)
        for task in (1, 2, 3):
            speeds = [command_profile(t, task, 0.5) for t in np.linspace(0, 20, 201)]
            self.assertGreater(max(speeds), 0.1)
            self.assertLessEqual(max(speeds), 0.5)
            self.assertEqual(command_profile(19, task, 0.5), 0.0)
        with self.assertRaises(ValueError):
            Config(max_speed=np.nan)
        with self.assertRaises(ValueError):
            Config(episode_seconds=0.0)

    def test_published_mechanism_and_virtual_work(self):
        result = validate_model()
        self.assertEqual(result["checks"], "passed")
        self.assertAlmostEqual(result["model"]["mass_kg"], OFFICIAL_MASS_KG)

    def test_velocity_body_axes(self):
        env = self.env
        env.reset(seed=1)
        velocity = [0.25, -0.13, 0.07]
        env.data.qvel[:3] = velocity
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env._body_velocity(), velocity, atol=1e-12)
        env.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(
            env._body_velocity(), [-0.13, -0.25, 0.07], atol=1e-12
        )

    def test_motor_envelope_and_wheel_sign(self):
        _, tau = decode_action(np.ones(6), DEFAULT_Q, np.zeros(6), "torque")
        np.testing.assert_array_equal(tau, [17, 17, -17, 17, 17, -17])
        _, fast = decode_action(
            np.ones(6), DEFAULT_Q, np.ones(6) * NO_LOAD_SPEED, "torque"
        )
        np.testing.assert_array_equal(fast, [0, 0, -17, 0, 0, -17])
        _, reverse = decode_action(
            -np.ones(6), DEFAULT_Q, -np.ones(6) * NO_LOAD_SPEED, "torque"
        )
        np.testing.assert_array_equal(reverse, [0, 0, 17, 0, 0, 17])

    def test_lifecycle_and_fault_latch(self):
        env = DiabloEnv(Config(episode_seconds=0.01))
        try:
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(6))
            env.reset(seed=1)
            _, _, _, truncated, _ = env.step(np.zeros(6))
            self.assertTrue(truncated)
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(6))
            self.assertTrue(
                env.model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_AUTORESET
            )
            env.reset(seed=1)
            with self.assertRaises(FloatingPointError):
                env._fault("test fault latch")
            with self.assertRaises(RuntimeError):
                env.reset()
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
