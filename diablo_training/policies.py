"""Stateless temporal actor-critic for the fixed Xingtian policy interface."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.utils import unpad_trajectories

from .model import WHEEL_HALF_TRACK

PROPRIO_DIM = 125
FRAME_DIM = 25
HISTORY_LENGTH = 5
HEIGHT_SCAN_DIM = 35
PRIVILEGED_DIM = 11


class _CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(channels)
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels_first = x.transpose(1, 2)
        channels_first = F.pad(
            channels_first,
            (2 * self.dilation, 0),
        )
        update = self.conv(channels_first).transpose(1, 2)
        return x + F.silu(self.norm(update))


class _ProprioEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.imu = nn.Sequential(
            nn.Linear(6, 32), nn.SiLU(), nn.LayerNorm(32)
        )
        self.command = nn.Sequential(
            nn.Linear(3, 16), nn.SiLU(), nn.LayerNorm(16)
        )
        self.leg = nn.Sequential(
            nn.Linear(4, 16), nn.SiLU(), nn.LayerNorm(16)
        )
        self.motor = nn.Sequential(
            nn.Linear(6, 24), nn.SiLU(), nn.LayerNorm(24)
        )
        self.previous_action = nn.Sequential(
            nn.Linear(6, 24), nn.SiLU(), nn.LayerNorm(24)
        )
        self.frame_fusion = nn.Sequential(
            nn.Linear(112, 96), nn.SiLU(), nn.LayerNorm(96)
        )
        self.temporal = nn.Sequential(
            _CausalBlock(96, dilation=1),
            _CausalBlock(96, dilation=2),
        )
        self.current_frame = nn.Sequential(
            nn.Linear(25, 64), nn.SiLU(), nn.LayerNorm(64)
        )
        self.fusion = nn.Sequential(
            nn.Linear(96 + 96 + 64 + 16, 192),
            nn.SiLU(),
            nn.LayerNorm(192),
        )
        self.film = nn.Linear(16, 384)

    def temporal_features(self, frames: torch.Tensor) -> torch.Tensor:
        groups = torch.cat(
            (
                self.imu(frames[..., 0:6]),
                self.command(frames[..., 6:9]),
                self.leg(frames[..., 9:13]),
                self.motor(frames[..., 13:19]),
                self.previous_action(frames[..., 19:25]),
            ),
            dim=-1,
        )
        return self.temporal(self.frame_fusion(groups))

    def forward(
        self, observations: torch.Tensor, raw_current: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        frames = observations.reshape(-1, HISTORY_LENGTH, FRAME_DIM)
        temporal = self.temporal_features(frames)
        command = self.command(frames[:, -1, 6:9])
        features = torch.cat(
            (
                temporal[:, -1],
                temporal.mean(dim=1),
                self.current_frame(raw_current),
                command,
            ),
            dim=-1,
        )
        features = self.fusion(features)
        gamma, beta = self.film(command).chunk(2, dim=-1)
        features = features * (1.0 + 0.1 * torch.tanh(gamma))
        features = features + 0.1 * torch.tanh(beta)
        return features, command


class _ResidualTrunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(192, 256), nn.SiLU(), nn.LayerNorm(256)
        )
        self.residual = nn.Sequential(
            nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 192)
        )
        self.output_norm = nn.LayerNorm(192)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input(x)
        return self.output_norm(x + self.residual(hidden))


class _ActorNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proprio = _ProprioEncoder()
        self.trunk = _ResidualTrunk()
        self.leg_head = nn.Sequential(
            nn.Linear(192, 64), nn.SiLU(), nn.Linear(64, 4)
        )
        self.wheel_head = nn.Sequential(
            nn.Linear(192 + 16 + 4, 64), nn.SiLU(), nn.Linear(64, 2)
        )
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        for head in (self.leg_head, self.wheel_head):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    @staticmethod
    def _wheel_features(raw_current: torch.Tensor) -> torch.Tensor:
        command_vx = raw_current[:, 6] / 10.0
        command_yaw = raw_current[:, 7] / 0.25
        wheel_forward = raw_current[:, 15:19:3] * 2.0
        mean_forward = wheel_forward.mean(dim=-1)
        estimated_yaw = (
            (wheel_forward[:, 1] - wheel_forward[:, 0])
            / (2.0 * WHEEL_HALF_TRACK)
        )
        return torch.stack(
            (
                command_vx,
                command_yaw,
                (command_vx - mean_forward) / 0.5,
                command_yaw - estimated_yaw,
            ),
            dim=-1,
        )

    def forward(
        self, observations: torch.Tensor, raw_current: torch.Tensor
    ) -> torch.Tensor:
        features, command = self.proprio(observations, raw_current)
        trunk = self.trunk(features)
        leg = self.leg_head(trunk)
        wheel = self.wheel_head(
            torch.cat(
                (trunk, command, self._wheel_features(raw_current)),
                dim=-1,
            )
        )
        # Network output order is fixed by the public action contract.
        return torch.cat((leg, wheel), dim=-1)[:, [0, 1, 4, 2, 3, 5]]


class _CriticNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proprio = _ProprioEncoder()
        self.height_scan = nn.Sequential(
            nn.Linear(HEIGHT_SCAN_DIM, 64),
            nn.SiLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.privileged = nn.Sequential(
            nn.Linear(PRIVILEGED_DIM, 32),
            nn.SiLU(),
            nn.LayerNorm(32),
            nn.Linear(32, 32),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(192 + 64 + 32, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(
        self, observations: torch.Tensor, raw_current: torch.Tensor
    ) -> torch.Tensor:
        proprio = observations[:, :PROPRIO_DIM]
        height_scan = observations[
            :, PROPRIO_DIM : PROPRIO_DIM + HEIGHT_SCAN_DIM
        ]
        privileged = observations[:, -PRIVILEGED_DIM:]
        features, _ = self.proprio(proprio, raw_current)
        return self.trunk(
            torch.cat(
                (
                    features,
                    self.height_scan(height_scan),
                    self.privileged(privileged),
                ),
                dim=-1,
            )
        )


class XingtianSequenceModel(MLPModel):
    """RSL-RL model with fixed actor input and no recurrent runtime state."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "swish",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims,
            activation,
            obs_normalization,
            distribution_cfg,
        )
        # The inherited MLP is not used; remove it so checkpoints only contain
        # parameters from the temporal residual graph.
        del self.mlp
        self.obs_set_name = obs_set
        if obs_set == "actor":
            if self.obs_dim != PROPRIO_DIM or output_dim != 6:
                raise ValueError(
                    "Actor must consume 125 observations and output 6 actions"
                )
            self.policy: nn.Module = _ActorNetwork()
        elif obs_set == "critic":
            expected = PROPRIO_DIM + HEIGHT_SCAN_DIM + PRIVILEGED_DIM
            if self.obs_dim != expected or output_dim != 1:
                raise ValueError(
                    f"Critic must consume {expected} observations and output 1"
                )
            self.policy = _CriticNetwork()
        else:
            raise ValueError(f"Unsupported observation set: {obs_set}")

    def _raw_observations(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat(
            [obs[group] for group in self.obs_groups],
            dim=-1,
        )

    def _forward_observations(self, observations: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if observations.ndim != 2 or observations.shape[-1] != self.obs_dim:
                raise ValueError(
                    f"Expected observations [batch,{self.obs_dim}], "
                    f"got {tuple(observations.shape)}"
                )
            if observations.dtype != torch.float32:
                raise TypeError(
                    f"Observations must be float32, got {observations.dtype}"
                )
        raw_current = observations[
            :, PROPRIO_DIM - FRAME_DIM : PROPRIO_DIM
        ]
        normalized = self.obs_normalizer(observations)
        return self.policy(normalized, raw_current)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        del hidden_state
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        output = self._forward_observations(self._raw_observations(obs))
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(output)
        return output

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxXingtianSequenceModel(self, verbose)


class _OnnxXingtianSequenceModel(nn.Module):
    is_recurrent = False

    def __init__(self, model: XingtianSequenceModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.model = copy.deepcopy(model)
        self.input_size = model.obs_dim
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.deterministic_output(
            self.model._forward_observations(observations)
        )

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
