"""B/C/D 阶段的地形奖励扩展。

为什么需要这个模块
------------------
A 阶段的对照实验表明：**仓库原奖励与验收指标不对齐**是端到端 RL 只拿到 2/7 的根因。

  PPO 微调后：奖励 −15 → +35（回合跑满、零跌倒），
  但前进航向偏 73°、横向漂移 1.09 m——从验收角度是灾难。

读 `LocomotionReward.evaluate` 可以发现三个具体缺口：

  1. **完全没有横向速度项**：`velocity[1]` 从头到尾没被用过，
     而 B/C/D 有 `max_lateral_y_m = 2.0` 的出界判定。
  2. **航向只有角速度、没有角度**：`yaw_tracking` 比较的是 `gyro[2]` vs 转向指令，
     只要不转就满分——航向漂到 73° 也拿满分。
  3. **站立惩罚太松**：`stand_still = 0.8·exp(−(v²+ω²)/0.01)`，
     0.032 m/s 的漂移只掉 8%（0.8 → 0.72），比 `forward_tracking` 的 1.2 小得多。

本模块用**新增任务类**的方式补这三项，不改任何共享文件（`xingtian.py` / `xingtian_env.py` /
`control.py` 一个字不动），因此不会再次触发全局指纹失效。

用法
----
配置里把 `environment_class` 指向本模块的 `XingtianTerrainRewardEnv` 即可，
配置类仍是 `XingtianStageConfig`。见 `examples/xingtian_stage_b_l1_wt.json`。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .xingtian import LocomotionReward, RewardWeights
from .xingtian_env import XingtianStageEnv


@dataclass(frozen=True)
class TerrainRewardWeights(RewardWeights):
    """在原有权重上加三个地形阶段需要的项。"""

    # 横向速度：B/C/D 有 2 m 出界判定，原奖励完全不惩罚
    lateral: float = 0.1
    lateral_scale: float = 0.2  # m/s，把 0.2 m/s 的横漂映射到 1.0 的惩罚

    # 航向保持：只在"不转向"的指令段生效（转向段不该被罚）
    # 0.3：地形上基础正项会塌到 ~0.1，罚项必须跟着降，否则整个奖励变负
    heading: float = 0.3
    heading_scale: float = 0.15  # rad ≈ 8.6°，正好是 ② 前进项的阈值量级

    # 站立残余速度：0.032 m/s → 约 0.15/秒（平地基准 0.028/步 ≈ 2.8/秒）
    stand_creep: float = 1.0
    stand_creep_scale: float = 0.01  # m²/s²，对应 0.1 m/s 量级

    # 终止惩罚软化的补偿量。
    # 原奖励 termination = −20，而前进一步只值 +0.012/步 → 一次摔倒要用 1600 步
    # （16 秒）的完美前进才能补回来，rough-l1 上不划算，于是策略学会"站着不动"。
    # 实测：原奖励跑 1M 步后前进只有 0.74 m（命令 0.5 m/s），转向却 PASS。
    # 设为 18.0 即把 −20 软化到 −2，让"冒险前进"有利可图。
    termination_relief: float = 18.0


def wrap_angle(angle: float) -> float:
    """把角度折到 (−π, π]。"""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


class TerrainCommandScheduler:
    """地形阶段专用指令调度器：短段 + 长恒定段混合。

    为什么需要它
    ------------
    `diablo evaluate`（训练分布）上，relief 策略 3/4 任务跑满 30 s；
    但 A 阶段验收剖面（恒定 0.5 m/s 连跑 14 s）上，8 个剖面全摔。

    原因是仓库自带的 `CommandScheduler` 每段只有 2–6 s、带 0.35 s 斜坡、
    频繁停车——**训练分布里根本不存在"长时间定速"这个条件**。
    于是策略在验收剖面上属于分布外，加多少步都学不会。

    本调度器 50% 概率发 10–15 s 的长恒定段（其中 60% 是定速直行），
    另一半仍是 2–6 s 的短段，保留切换与响应能力。

    接口与 `CommandScheduler` 一致（`command_at(sim_seconds)`），
    因此可以直接替换，不改任何共享文件。
    """

    def __init__(
        self,
        rng: np.random.Generator,
        max_forward: float = 0.5,
        max_backward: float = 0.4,
        max_yaw_rate: float = 1.0,
        height: float = 0.25,
        ramp_seconds: float = 0.35,
        long_probability: float = 0.5,
        long_duration: tuple[float, float] = (10.0, 15.0),
        short_duration: tuple[float, float] = (2.0, 6.0),
        stop_probability: float = 0.15,
    ):
        self.rng = rng
        self.max_forward = max_forward
        self.max_backward = max_backward
        self.max_yaw_rate = max_yaw_rate
        self.height = height
        self.ramp_seconds = ramp_seconds
        self.long_probability = long_probability
        self.long_duration = long_duration
        self.short_duration = short_duration
        self.stop_probability = stop_probability

        self.started_at = 0.0
        self.previous = np.array([0.0, 0.0, height])
        self.target, self.duration = self._sample()
        self._next_at = self.duration

    def _sample(self) -> tuple[np.ndarray, float]:
        if self.rng.random() < self.long_probability:
            duration = float(self.rng.uniform(*self.long_duration))
            # 长段偏向"定速直行"，直接对应 A 阶段的验收剖面
            if self.rng.random() < 0.6:
                speed = float(self.rng.uniform(0.3, self.max_forward))
                if self.rng.random() < 0.25:
                    speed = -float(self.rng.uniform(0.2, self.max_backward))
                return np.array([speed, 0.0, self.height]), duration
        else:
            duration = float(self.rng.uniform(*self.short_duration))

        if self.rng.random() < self.stop_probability:
            return np.array([0.0, 0.0, self.height]), duration
        forward = float(
            self.rng.uniform(-self.max_backward, self.max_forward)
        )
        yaw = float(self.rng.uniform(-self.max_yaw_rate, self.max_yaw_rate))
        if self.rng.random() < 0.3:
            yaw = 0.0
        return np.array([forward, yaw, self.height]), duration

    def command_at(self, sim_seconds: float) -> np.ndarray:
        guard = 0
        while sim_seconds >= self._next_at and guard < 1000:
            self.previous = self.target
            self.started_at = self._next_at
            self.target, self.duration = self._sample()
            self._next_at = self.started_at + self.duration
            guard += 1
        alpha = min(1.0, (sim_seconds - self.started_at) / self.ramp_seconds)
        return (1.0 - alpha) * self.previous + alpha * self.target


class TerrainReward(LocomotionReward):
    """在 `LocomotionReward` 基础上追加三项，其余各项原样保留。"""

    def __init__(self, weights: TerrainRewardWeights | None = None):
        super().__init__(weights or TerrainRewardWeights())
        self.heading_anchor = 0.0

    @property
    def w(self) -> TerrainRewardWeights:
        return self.weights  # type: ignore[return-value]

    def evaluate(
        self,
        transition: dict,
        action: np.ndarray,
        previous_action: np.ndarray,
        *,
        dt: float = 0.01,
        previous_difference: np.ndarray | None = None,
    ):
        breakdown = super().evaluate(
            transition,
            action,
            previous_action,
            dt=dt,
            previous_difference=previous_difference,
        )
        w = self.w
        command = np.asarray(transition["command"], dtype=float)
        velocity = np.asarray(transition["velocity"], dtype=float)
        yaw = float(transition.get("yaw", 0.0))

        # 注意：基类的十项终值都乘了 dt（单步奖励 ≈ 0.028，即 2.8/秒），
        # 所以这里新增的三项也必须乘 dt，否则会强 100 倍、把其余信号全压死。
        # 自检实测：不乘 dt 时"理想直行"只有 0.028，而航向项一项就有 −0.97。

        # ① 横向速度（原奖励完全没有这一项）
        breakdown.terms["lateral_velocity"] = -w.lateral * float(
            (velocity[1] / w.lateral_scale) ** 2
        ) * dt

        # ② 航向保持：只在直行段生效；用 tanh 饱和，避免大误差时奖励爆炸
        if abs(command[1]) < 0.05:
            error = wrap_angle(yaw - self.heading_anchor)
            breakdown.terms["heading_hold"] = -w.heading * float(
                np.tanh((error / w.heading_scale) ** 2)
            ) * dt

        # ③ 站立残余速度：二次惩罚，0.032 m/s → −0.51/秒，0.1 m/s → −5.0/秒
        if max(abs(command[0]), abs(command[1])) < 0.01:
            creep = float(velocity[0] ** 2 + velocity[1] ** 2)
            breakdown.terms["stand_creep"] = (
                -w.stand_creep * creep / w.stand_creep_scale * dt
            )

        # ④ 终止惩罚软化：基类在 terminated 时加 −20，这里补回 termination_relief
        if transition.get("terminated", False):
            breakdown.terms["termination_relief"] = w.termination_relief

        return breakdown


class XingtianTerrainRewardEnv(XingtianStageEnv):
    """B/C/D 阶段环境：地形照旧，奖励换成 `TerrainReward`。

    只在 `reward_for_transition` 里注入 `yaw` 与航向锚点，其余逻辑全部走父类，
    保证与 `XingtianStageEnv` 的物理、观测、终止条件逐位一致。
    """

    def __init__(self, config=None, reward_weights=None, render_mode=None):
        super().__init__(config, reward_weights, render_mode)
        self.terrain_reward = TerrainReward(reward_weights)
        self.reward_function = self.terrain_reward
        self._last_yaw_command = 0.0
        self.terrain_scheduler = None

    def _yaw(self) -> float:
        w, x, y, z = self.data.xquat[self.base_id]
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    def get_command(self, sim_seconds: float) -> np.ndarray:
        """替换仓库调度器：加入 10–15 s 长恒定段，覆盖验收剖面。"""
        if self.terrain_scheduler is None:
            cfg = self.xingtian_config
            self.terrain_scheduler = TerrainCommandScheduler(
                self.np_random,
                max_forward=cfg.max_speed,
                max_backward=cfg.max_backward_speed,
                max_yaw_rate=cfg.max_yaw_rate,
                height=0.5 * (cfg.min_height + cfg.max_height),
                ramp_seconds=cfg.command_ramp_seconds,
            )
        return self.terrain_scheduler.command_at(sim_seconds)

    def reset(self, seed=None, options=None):
        self.terrain_scheduler = None
        observation, info = super().reset(seed=seed, options=options)
        self.terrain_reward.heading_anchor = self._yaw()
        self._last_yaw_command = 0.0
        return observation, info

    def reward_for_transition(self, transition, action, previous_action):
        transition = dict(transition)
        yaw = self._yaw()
        command = np.asarray(transition["command"], dtype=float)

        # 从"转向"切回"直行"的那一刻，把当前航向设为锚点
        if abs(command[1]) < 0.05 <= abs(self._last_yaw_command):
            self.terrain_reward.heading_anchor = yaw
        self._last_yaw_command = float(command[1])

        transition["yaw"] = yaw
        return super().reward_for_transition(transition, action, previous_action)
