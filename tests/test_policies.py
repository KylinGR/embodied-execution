"""CPU tests for the fixed-interface temporal residual policy."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

try:
    import torch
    from tensordict import TensorDict

    from diablo_training.policies import (
        HEIGHT_SCAN_DIM,
        PRIVILEGED_DIM,
        PROPRIO_DIM,
        XingtianSequenceModel,
    )

    DEPENDENCIES_AVAILABLE = True
except ModuleNotFoundError:
    DEPENDENCIES_AVAILABLE = False


@unittest.skipUnless(
    DEPENDENCIES_AVAILABLE,
    "PyTorch, TensorDict for RSL-RL, or the policy dependencies are unavailable",
)
class PolicyTests(unittest.TestCase):
    obs_groups = {"actor": ["actor"], "critic": ["critic"]}

    def make_observations(self, batch: int = 3) -> TensorDict:
        critic_dim = PROPRIO_DIM + HEIGHT_SCAN_DIM + PRIVILEGED_DIM
        return TensorDict(
            {
                "actor": torch.zeros(batch, PROPRIO_DIM),
                "critic": torch.zeros(batch, critic_dim),
            },
            batch_size=[batch],
        )

    def make_actor(self, obs: TensorDict) -> XingtianSequenceModel:
        return XingtianSequenceModel(
            obs,
            self.obs_groups,
            "actor",
            6,
            hidden_dims=(256, 256),
            activation="swish",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.3,
                "std_type": "scalar",
            },
        )

    def test_actor_and_critic_keep_the_public_io_contract(self):
        obs = self.make_observations()
        actor = self.make_actor(obs)
        critic = XingtianSequenceModel(
            obs,
            self.obs_groups,
            "critic",
            1,
            hidden_dims=(256, 256),
            activation="swish",
            obs_normalization=True,
        )
        action = actor(obs)
        value = critic(obs)
        self.assertEqual(action.shape, (3, 6))
        self.assertEqual(action.dtype, torch.float32)
        self.assertTrue(torch.count_nonzero(action) == 0)
        self.assertEqual(value.shape, (3, 1))
        self.assertEqual(value.dtype, torch.float32)
        self.assertLess(sum(p.numel() for p in actor.parameters()), 600_000)
        self.assertFalse(actor.is_recurrent)
        self.assertFalse(critic.is_recurrent)
        torch.testing.assert_close(
            actor.distribution.std_param,
            torch.full((6,), 0.3),
        )
        self.assertTrue(
            all(not key.startswith("mlp.") for key in actor.state_dict())
        )

    def test_temporal_encoder_is_causal(self):
        obs = self.make_observations(batch=2)
        actor = self.make_actor(obs)
        frames = torch.randn(2, 5, 25)
        changed = frames.clone()
        changed[:, -1] += 3.0
        baseline = actor.policy.proprio.temporal_features(frames)
        modified = actor.policy.proprio.temporal_features(changed)
        torch.testing.assert_close(
            baseline[:, :-1],
            modified[:, :-1],
            atol=0.0,
            rtol=0.0,
        )
        self.assertGreater(
            torch.abs(baseline[:, -1] - modified[:, -1]).max().item(),
            0.0,
        )

    def test_leg_and_wheel_heads_follow_public_action_order(self):
        obs = self.make_observations(batch=1)
        actor = self.make_actor(obs)
        with torch.no_grad():
            actor.policy.leg_head[-1].bias.copy_(
                torch.tensor([1.0, 2.0, 3.0, 4.0])
            )
            actor.policy.wheel_head[-1].bias.copy_(
                torch.tensor([5.0, 6.0])
            )
        output = actor(obs)
        torch.testing.assert_close(
            output,
            torch.tensor([[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]]),
        )

    def test_actor_state_round_trips_without_shape_changes(self):
        obs = self.make_observations(batch=1)
        source = self.make_actor(obs)
        target = self.make_actor(obs)
        target.load_state_dict(source.state_dict(), strict=True)
        for source_value, target_value in zip(
            source.state_dict().values(),
            target.state_dict().values(),
            strict=True,
        ):
            torch.testing.assert_close(source_value, target_value)

    def test_onnx_wrapper_has_one_input_and_output(self):
        obs = self.make_observations(batch=4)
        actor = self.make_actor(obs)
        exported = actor.as_onnx(False)
        self.assertEqual(exported.input_names, ["obs"])
        self.assertEqual(exported.output_names, ["actions"])
        dummy = exported.get_dummy_inputs()
        self.assertEqual(len(dummy), 1)
        self.assertEqual(dummy[0].shape, (1, 125))
        output = exported(torch.zeros(4, 125))
        self.assertEqual(output.shape, (4, 6))
        self.assertEqual(output.dtype, torch.float32)

        if importlib.util.find_spec("onnx") is not None:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "policy.onnx"
                torch.onnx.export(
                    exported,
                    dummy,
                    path,
                    input_names=exported.input_names,
                    output_names=exported.output_names,
                    dynamic_axes={
                        "obs": {0: "batch"},
                        "actions": {0: "batch"},
                    },
                    opset_version=18,
                    dynamo=False,
                )
                self.assertGreater(path.stat().st_size, 0)
                import onnx

                exported_model = onnx.load(str(path))
                input_shape = exported_model.graph.input[0].type.tensor_type.shape
                output_shape = exported_model.graph.output[0].type.tensor_type.shape
                self.assertEqual(input_shape.dim[-1].dim_value, 125)
                self.assertEqual(output_shape.dim[-1].dim_value, 6)
                self.assertEqual(input_shape.dim[0].dim_param, "batch")
                self.assertEqual(output_shape.dim[0].dim_param, "batch")

                if importlib.util.find_spec("onnxruntime") is not None:
                    import onnxruntime as ort

                    sample = torch.randn(4, 125)
                    expected = exported(sample).detach().numpy()
                    session = ort.InferenceSession(
                        str(path),
                        providers=["CPUExecutionProvider"],
                    )
                    actual = session.run(
                        None,
                        {session.get_inputs()[0].name: sample.numpy()},
                    )[0]
                    torch.testing.assert_close(
                        torch.from_numpy(actual),
                        torch.from_numpy(expected),
                        atol=1e-5,
                        rtol=1e-5,
                    )
