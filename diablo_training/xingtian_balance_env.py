"""A 阶段解析平衡任务：腿由关节 PD 固定，轮子由解析状态反馈闭环。

为什么单独做成一个任务类，而不是用环境变量切：
    1. `policy_contract` 会把「任务类源文件」和 xingtian*.py 一起算指纹。
       控制器参数放在本文件里 = 参数进契约；换一组增益就是换一个契约，
       不会出现「同一个 sha、两种行为」。
    2. `resolved.json` 里本来就有 `environment_class` 这个扩展点，
       diag / evaluate / train 全走它，不需要改 env.py。
    3. 纯 RL 的 `XingtianFlatEnv` 一行不动，旧 checkpoint 仍可原样加载，
       A 阶段的 RL 对照结果仍然可复现。

域随机化（BalanceConfig.randomization，仅本任务支持）：
    mass      质量倍率，如 [0.95, 1.05]
    friction  地面滑动摩擦倍率，如 [0.8, 1.2]
    torque    力矩倍率（模拟电机常数/电压误差），如 [0.9, 1.1]
    delay_ms  控制延迟（毫秒），如 [10, 20]；每回合在区间内均匀采样
    randomization 为 null 时不扰动，行为与 analytic-v4 逐位相同。

用法：
    python -m diablo_training.xingtian_diag2 \
        --model <任意占位权重> \
        --resolved examples/xingtian_balance_analytic.json \
        --seeds 2002 2003 2004 --out runs/diag/xxx
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import balance
from .model import CONTROL_DT, DEFAULT_Q
from .xingtian_env import XingtianConfig, XingtianFlatEnv

# ---------------------------------------------------------------- 控制参数
# 注意：这些数在 A 阶段 analytic-v3 上验过（24/24）。改一个数就等于换契约。
STAND = dict(k_th=30.0, k_thd=6.0, k_x=60.0, k_v=30.0, ki_th=1.5)
MOVE = dict(k_th=30.0, k_thd=6.0, k_x=10.0, k_v=45.0, ki_th=1.5)
YAW = dict(k_w=2.0, k_psi=2.0, clamp=2.5)
# 制动段：k_th ×2.5、k_v 75、上限 17、冻结俯仰积分、关位置项
#   实测停车 0.0967 → 0.0649 m；俯仰最低 -20.2°，距跌倒判据还有 5.6°
STOP = dict(
    k_th_scale=2.5,
    k_thd_scale=1.0,
    k_x_scale=0.0,
    k_v=75.0,
    ki_scale=0.0,
    clamp=17.0,
    pos_limit=99.0,
    decel=0.0,
    exit_v=0.10,
)
TAU_UP = 0.35          # 起步滤波；0.25 时前向超调 31%
TAU_DOWN = 0.03        # 减速滤波
STOP_THRESHOLD = 0.02  # |vx 指令| 低于此值算停车
STOP_ANCHOR_EACH = 0.0 # 只在停车那一刻钉一次行程锚点
TORQUE_CLAMP = 12.0
I_TH_LIMIT = 0.6

# 延迟补偿默认值（毫秒）；实际由 BalanceConfig.delay_comp_ms 覆盖，便于扫档
DELAY_COMP_MS = 0.0
# 无记忆模式（关掉积分器），给行为克隆当 teacher 用
MARKOV_ONLY = 0.0

# 腿部 PD（固定到 DEFAULT_Q）
LEG_KP = 60.0
LEG_KD = 2.0

_RANDOM_KEYS = ("mass", "friction", "torque", "delay_ms")


@dataclass
class BalanceConfig(XingtianConfig):
    """在 XingtianConfig 上只加两个字段：随机化范围、延迟补偿量。"""

    randomization: dict[str, Any] | None = None
    delay_comp_ms: float = 0.0
    markov_only: bool = False
    stand_kv: float | None = None
    move_kv: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if not np.isfinite(self.delay_comp_ms) or self.delay_comp_ms < 0:
            raise ValueError("delay_comp_ms must be finite and non-negative")
        rand = self.randomization
        if rand is None:
            return
        if not isinstance(rand, dict):
            raise ValueError("randomization must be an object or null")
        unknown = set(rand) - set(_RANDOM_KEYS)
        if unknown:
            raise ValueError(f"unknown randomization keys: {sorted(unknown)}")
        for key, value in rand.items():
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"randomization[{key!r}] must be [low, high]")
            low, high = float(value[0]), float(value[1])
            if not np.isfinite([low, high]).all() or low > high:
                raise ValueError(f"randomization[{key!r}] has invalid range")


def _install_params() -> None:
    """把本文件的参数装进 balance 模块。"""
    for name, value in (("STAND", STAND), ("MOVE", MOVE), ("YAW", YAW), ("STOP", STOP)):
        target = getattr(balance, name, None)
        if target is None:
            setattr(balance, name, dict(value))
        else:
            target.clear()
            target.update(value)
    for name, value in (
        ("TAU_UP", TAU_UP), ("TAU_DOWN", TAU_DOWN),
        ("STOP_THRESHOLD", STOP_THRESHOLD), ("STOP_ANCHOR_EACH", STOP_ANCHOR_EACH),
        ("TORQUE_CLAMP", TORQUE_CLAMP), ("I_TH_LIMIT", I_TH_LIMIT),
        ("TH_PREDICT_S", DELAY_COMP_MS / 1000.0), ("MARKOV_ONLY", MARKOV_ONLY),
    ):
        setattr(balance, name, value)


_install_params()


class XingtianBalanceEnv(XingtianFlatEnv):
    """解析平衡任务：轮子由 balance 闭环，腿保持固定姿态。"""

    def __init__(self, config=None, reward_weights=None, render_mode=None):
        super().__init__(config, reward_weights, render_mode)
        self._balance_bias = np.zeros(6)
        self._bal_state = None
        self._bias_ring: list[np.ndarray] = []
        self._delay_steps = 0
        self._torque_scale = 1.0
        # 延迟补偿量由 config 决定（毫秒 → 秒）
        balance.TH_PREDICT_S = float(getattr(self.config, "delay_comp_ms", 0.0)) / 1000.0
        # 无记忆模式：给行为克隆当 teacher 时打开
        balance.MARKOV_ONLY = 1.0 if getattr(self.config, "markov_only", False) else 0.0
        # 无记忆模式下位置积分没了，靠提高速度增益压稳态误差
        # （实测 stand 漂移 0.062 → 0.0035，forward 速度误差 5.2% → 5.0%）
        if getattr(self.config, "stand_kv", None) is not None:
            balance.STAND["k_v"] = float(self.config.stand_kv)
        if getattr(self.config, "move_kv", None) is not None:
            balance.MOVE["k_v"] = float(self.config.move_kv)
        # 名义值快照：每回合先把模型恢复成名义值，再采样扰动
        self._nominal_mass = self.model.body_mass.copy()
        self._nominal_inertia = self.model.body_inertia.copy()
        self._nominal_friction = self.model.geom_friction.copy()

    # ------------------------------------------------------------ 随机化
    def _restore_model(self) -> None:
        self.model.body_mass[:] = self._nominal_mass
        self.model.body_inertia[:] = self._nominal_inertia
        self.model.geom_friction[:] = self._nominal_friction
        self._torque_scale = 1.0
        self._delay_steps = 0
        self._bias_ring = []

    def _sample_randomization(self, seed: int | None) -> None:
        rand = getattr(self.config, "randomization", None)
        if not rand:
            return
        rng = np.random.default_rng(seed)
        if "mass" in rand:
            low, high = rand["mass"]
            scale = float(rng.uniform(low, high))
            # world 不动；质量与惯量同比缩放，保持密度一致
            self.model.body_mass[1:] *= scale
            self.model.body_inertia[1:] *= scale
        if "friction" in rand:
            low, high = rand["friction"]
            self.model.geom_friction[:, 0] *= float(rng.uniform(low, high))
        if "torque" in rand:
            low, high = rand["torque"]
            self._torque_scale = float(rng.uniform(low, high))
        if "delay_ms" in rand:
            low, high = rand["delay_ms"]
            delay_ms = float(rng.uniform(low, high))
            self._delay_steps = int(round(delay_ms / 1000.0 / CONTROL_DT))
        # 注意：这里**不能**调 mujoco.mj_setConst(model, data)。
        # 它会把 data 重置成 qpos0（URDF 默认姿态，base 高度 0.400 m），
        # 于是机器人从 0.4 m 处自由落体、回合 0.3 s 就结束（实测过）。
        # 质量/惯量 mj_step 每步直接读，不重算派生常数只会让约束权重有小偏差。

    # ------------------------------------------------------------ 生命周期
    def reset(self, *, seed=None, options=None):
        self._restore_model()
        result = super().reset(seed=seed, options=options)
        self._bal_state = None
        self._bias_ring = []
        self._sample_randomization(seed)
        return result

    # ------------------------------------------------------------ env.py 里已有的两个钩子
    def _update_balance_control(self):
        if self._bal_state is None:
            self._bal_state = balance.make_state(self.data, self.base_id)
        bias = balance.compute(
            self._bal_state,
            self.data,
            self.model,
            self.base_id,
            self.command,
            CONTROL_DT,
        ) * self._torque_scale
        if self._delay_steps <= 0:
            self._balance_bias = bias
            return
        # 控制延迟：现在算出来的力矩，delay_steps 个控制周期之后才生效
        self._bias_ring.append(bias)
        if len(self._bias_ring) > self._delay_steps:
            self._balance_bias = self._bias_ring.pop(0)
        else:
            self._balance_bias = np.zeros(6)

    def _apply_control(self, applied):
        out = np.array(applied, dtype=float, copy=True)
        q = self.data.qpos[self.qi]
        dq = self.data.qvel[self.vi]
        for j in (0, 1, 3, 4):
            out[j] = LEG_KP * (DEFAULT_Q[j] - q[j]) - LEG_KD * dq[j]
        out[2] = 0.0
        out[5] = 0.0
        return out * self._torque_scale


# 便于 diag/eval 直接引用
ENVIRONMENT = "diablo_training.xingtian_balance_env:XingtianBalanceEnv"
CONFIG = "diablo_training.xingtian_balance_env:BalanceConfig"
