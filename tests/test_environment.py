"""Physics/interface checks that do not require a trained policy or a GPU."""

import unittest

import numpy as np
import mujoco

from diablo_training.env import Config, DiabloEnv, command_profile
from diablo_training.model import JOINTS
from diablo_training.control import decode_action
from diablo_training.model import DEFAULT_Q, NO_LOAD_SPEED, STALL_TORQUE
from diablo_training.validation import validate_model
from diablo_training.contract import policy_contract, require_compatible


class EnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = DiabloEnv()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_sensor_and_actuator_contract(self):
        env = self.env
        observation, _ = env.reset(seed=9)
        self.assertEqual(env.model.nu, 6)
        self.assertEqual(observation.shape, (125,))
        self.assertTrue(env.observation_space.contains(observation))
        names = [env.model.joint(int(j)).name for j in env.model.actuator_trnid[:, 0]]
        self.assertEqual(names, list(JOINTS))
        self.assertEqual(env.data.ncon, 0, "Initial pose must not penetrate the ground")
        self.assertAlmostEqual(np.linalg.norm(observation[-22:-19]), 1.0, places=5)

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

    def test_courses_reach_motion_and_then_hold(self):
        speed = 0.5
        expected = {
            0: 0.0,  # stand_balance：全程零指令
            1: 0.0,  # single_start_stop：10 秒后停车保持
            2: 0.5,  # cruise_balance：持续巡航，1.0 × speed
            3: 0.2,  # drive_accel_decel：减速到低速并保持，0.4 × speed
            4: 0.0,  # repeat_start_stop：16 秒后停车保持
        }
        for task, value in expected.items():
            self.assertAlmostEqual(command_profile(19, task, speed), value)


    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            Config(max_speed=np.nan)
        with self.assertRaises(ValueError):
            Config(episode_seconds=0.0)

    def test_published_mechanism_and_virtual_work(self):
        self.assertEqual(validate_model()["checks"], "passed")

    def test_velocity_is_in_body_axes_not_inertial_principal_axes(self):
        env = self.env

        # 这个测试检查世界坐标到机体坐标的旋转，
        # 因此关闭随机初始 yaw，从确定的对齐姿态开始。
        env.reset(seed=1,options={"randomize_init": False},)
        velocity = [0.25, -0.13, 0.07]

        # 清除 reset 可能产生的角速度和关节速度
        env.data.qvel[:] = 0.0
        env.data.qvel[:3] = velocity

        mujoco.mj_forward(env.model, env.data)
        np.testing.assert_allclose(env._body_velocity(), velocity, atol=1e-12)

        # 测试绕Z轴旋转 45 度的情况，检查机体坐标是否正确旋转。
        env.data.qpos[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
        mujoco.mj_forward(env.model, env.data)

        np.testing.assert_allclose(
            env._body_velocity(), [-0.13, -0.25, 0.07], atol=1e-12
        )

    def test_command_history_and_reward_share_transition_clock(self):
        class ClockEnv(DiabloEnv):
            def get_command(self, sim_seconds):
                return np.array([sim_seconds, 0.0, self.reference_height])

            def reward_for_transition(self, transition, action, previous_action):
                return transition["command"][0]

        env = ClockEnv()
        try:
            initial, _ = env.reset(seed=1)
            observation, reward, _, _, info = env.step(np.ones(6) * 0.1)
            np.testing.assert_array_equal(observation[:100], initial[25:])
            self.assertAlmostEqual(reward, 0.0)
            self.assertAlmostEqual(info["command_mps"], 0.0)
            self.assertAlmostEqual(observation[-19], 0.01 * 10, places=6)
            np.testing.assert_allclose(observation[-6:], 0.1)
            self.assertAlmostEqual(env.data.time, 0.01)
        finally:
            env.close()

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

    def test_lifecycle_and_checkpoint_compatibility(self):
        env = DiabloEnv(Config(episode_seconds=0.01))
        try:
            with self.assertRaises(RuntimeError):
                env.step(np.zeros(6))
            env.reset(seed=1)
            saved = {"contract": policy_contract(env)}
            require_compatible(saved, env)
            with self.assertRaises(ValueError):
                require_compatible({}, env)
            saved["contract"]["sha256"] = "different"
            with self.assertRaises(ValueError):
                require_compatible(saved, env)
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
