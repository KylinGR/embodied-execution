"""门控双专家（简化 MoE）特征提取器。

要解决的问题：
    七次调奖励/观测都失败，症状一致——改一处就丢一处：
        前进加速要"轮速猛"，原地转向要"左右反相"，姿态稳定要"力矩保守"。
    这三类需求在同一个 MLP 里共用同一份梯度，互相拉扯（gradient conflict）。

参考：
    MoE-Loco: Mixture of Experts for Multitask Locomotion (arXiv 2503.08564)
    "Using MoE, we mitigate the gradient conflicts that typically arise in
     multitask reinforcement learning."

设计（最小改动版）：
    obs[125] → 共享主干 [128→64] → ┬─ 前进专家 ─┐
                                    └─ 转向专家 ─┴→ 特征[64] → pi/vf → action[6]
                      门控：|obs[107]| > 1.0 走转向专家

    · 门控只依赖观测（指令就在 obs 里），所以整个 MoE 可以放进 features_extractor
      —— 不需要改仓库的 env / 奖励 / 训练脚本
    · obs 125 维、action 6 维完全不变
    · torch.where 做硬选择：每个专家只从自己的样本得到梯度

观测布局（已确认）：
    history (5, 25) → flatten() → [帧0(25) | 帧1(25) | ... | 帧4(25)]，帧4 最新
    帧内：gyro(0-2) gravity(3-5) cmd(6=vx, 7=yaw, 8=height) q(9-12) dq(13-18) prev_action(19-24)
    全局索引：cmd_yaw = 4*25 + 7 = 107

环境变量：
    MOE_YAW_INDEX   偏航指令在 obs 里的索引（默认 107）
    MOE_GATE        门控阈值（默认 1.0，对应 cmd_yaw = 0.1，因为观测系数是 10）
    MOE_FEATURES    输出特征维度（默认 64）
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

OBS_DIM = 125
CMD_YAW_INDEX = int(os.environ.get("MOE_YAW_INDEX", "107"))
GATE_THRESHOLD = float(os.environ.get("MOE_GATE", "1.0"))
FEATURES_DIM = int(os.environ.get("MOE_FEATURES", "64"))

# 最新一帧的 3 个指令维（vx, yaw, height）
CMD_SLICE = slice(4 * 25 + 6, 4 * 25 + 9)


class MoEFeatures(BaseFeaturesExtractor):
    """共享主干 + 两个按偏航指令硬门控的专家。"""

    def __init__(self, observation_space: spaces.Box, features_dim: int = FEATURES_DIM):
        super().__init__(observation_space, features_dim)
        if observation_space.shape != (OBS_DIM,):
            raise ValueError(
                f"MoEFeatures 期望观测维度 {OBS_DIM}，实际 {observation_space.shape}"
            )

        self.yaw_index = CMD_YAW_INDEX
        self.gate_threshold = GATE_THRESHOLD
        self._gate_counts = [0, 0]
        self._count_calls = 0

        self.trunk = nn.Sequential(
            nn.Linear(OBS_DIM, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        # 专家输入：主干特征 + 最新的 3 个指令维，让它知道当前要干什么
        expert_in = 64 + 3
        self.expert_forward = nn.Sequential(
            nn.Linear(expert_in, 64),
            nn.Tanh(),
        )
        self.expert_turn = nn.Sequential(
            nn.Linear(expert_in, 64),
            nn.Tanh(),
        )
        self.project = nn.Sequential(
            nn.Linear(64, features_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations.float()
        trunk = self.trunk(obs)
        cmd = obs[:, CMD_SLICE]

        expert_input = torch.cat([trunk, cmd], dim=1)

        gate = obs[:, self.yaw_index].abs() > self.gate_threshold  # (batch,) bool

        # 门控统计：只在最开始统计前 200 次，避免每步 .item() 同步 GPU 拖慢训练
        if self._count_calls < 200:
            self._count_calls += 1
            with torch.no_grad():
                self._gate_counts[0] += int((~gate).sum().item())
                self._gate_counts[1] += int(gate.sum().item())

        forward_out = self.expert_forward(expert_input)
        turn_out = self.expert_turn(expert_input)
        mask = gate.unsqueeze(1)
        # torch.where：梯度只回到被选中的那个专家
        mixed = torch.where(mask, turn_out, forward_out)
        return self.project(mixed)

    def gate_usage(self) -> tuple[int, int]:
        """(前进专家样本数, 转向专家样本数)，用于确认门控真的在分流。"""
        return self._gate_counts[0], self._gate_counts[1]
