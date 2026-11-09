"""把解析控制器当作 teacher，训一个只用 125 维观测的 student。

为什么要做这件事：
    teacher（解析控制器）用的是**真值**：机体线速度 v、世界位置积分 s。
    student 只有 125 维本体感觉。如果 student 能逼近 teacher，说明这些量在
    观测里是可推断的（轮速 + 陀螺 + 历史）；如果逼近不了，就是信息天花板，
    再调 RL 奖励也没用。

做法：
    teacher 的力矩 = 腿 PD(DEFAULT_Q) + 轮子解析偏置
    student 在 action_mode="torque" 的环境里学同样的动作
    动作换算：torque 模式下 action = τ / (STALL_TORQUE · OBSERVATION_SIGNS)

三个子命令：
    smoke   只跑 1 个回合，打印形状和力矩范围（先跑这个）
    collect 用 teacher 采数据，存 npz
    dagger  DAgger：student 自己跑，偏到哪就在哪问 teacher 要标签，反复重训
    train   只用已有 npz 做行为克隆（不跑仿真，调参快）

用法：
    # 0) 先验证管线
    python -m diablo_training.distill_balance smoke --student-config examples/xingtian_balance_student.json

    # 1) 采数据（teacher 很快，10 回合约 1 分钟）
    python -m diablo_training.distill_balance collect --student-config examples/xingtian_balance_student.json \
        --episodes 10 --out runs/distill/data.npz

    # 2) DAgger（3 轮，每轮 5 回合）
    python -m diablo_training.distill_balance dagger --student-config examples/xingtian_balance_student.json \
        --episodes 5 --iterations 3 --out runs/distill

    # 3) 评测（用官方诊断）
    python -m diablo_training.xingtian_diag2 --model runs/distill/policy.zip \
        --resolved examples/xingtian_balance_student.json --seeds 2002 2003 2004 --out runs/diag/student-v1
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import balance
from . import xingtian_balance_env as _balance_env  # noqa: F401  导入即安装控制器参数
from .model import CONTROL_DT, DEFAULT_Q, LEGS, OBSERVATION_SIGNS, STALL_TORQUE, WHEELS
from .xingtian_diag2 import build_env

LEG_KP = 60.0
LEG_KD = 2.0
TORQUE_SCALE = STALL_TORQUE * OBSERVATION_SIGNS

# position_velocity 模式的反解系数（见 control.decode_action）
PV_A = np.array([15.0, 18.0, 0.0, 15.0, 18.0, 0.0])
PV_KD = np.array([0.3, 0.2, 0.3, 0.3, 0.2, 0.3])
PV_WHEEL_GAIN = 3.0  # 0.3 * 10


# ---------------------------------------------------------------- teacher

def teacher_torque(env, bstate) -> np.ndarray:
    """解析控制器此刻会给出的 6 维力矩（腿 PD + 轮子偏置）。"""
    bias = balance.compute(
        bstate, env.data, env.model, env.base_id, env.command, CONTROL_DT
    )
    q = env.data.qpos[env.qi]
    dq = env.data.qvel[env.vi]
    tau = np.zeros(6)
    for j in LEGS:
        tau[j] = LEG_KP * (DEFAULT_Q[j] - q[j]) - LEG_KD * dq[j]
    tau[WHEELS] = bias[WHEELS]
    return tau


def to_action(tau: np.ndarray) -> np.ndarray:
    """torque 模式：action 与力矩成正比。"""
    return np.clip(tau / TORQUE_SCALE, -1.0, 1.0)


def to_action_pv(tau: np.ndarray, q: np.ndarray, dq: np.ndarray) -> np.ndarray:
    """position_velocity 模式的反解：把期望力矩换算回动作。

    decode_action 在 pv 模式下：
        τ = a·(DEFAULT_Q + 0.5·act − q) − kd·dq + ff      （腿）
        τ = −3·act + ff                                    （轮）
    反解得 act；腿的目标动作接近 0（因为 teacher 的腿就是 PD 到 DEFAULT_Q），
    轮子承载全部平衡信息。
    """
    from .control import radial_feedforward

    ff = radial_feedforward(q)
    act = np.zeros(6)
    for j in LEGS:
        act[j] = 2.0 * ((tau[j] + PV_KD[j] * dq[j] - ff[j]) / PV_A[j]
                        - DEFAULT_Q[j] + q[j])
    for j in WHEELS:
        # 轮子没有位置项（a=0），实际是：τ = -3·act - 0.3·dq
        # 漏掉 -0.3·dq 会让标签随轮速线性漂移（站立 0、前进时偏 0.5），
        # 表现就是"能站不能走"。
        act[j] = (ff[j] - tau[j] - PV_KD[j] * dq[j]) / PV_WHEEL_GAIN
    return np.clip(act, -1.0, 1.0)


def to_action_wt(tau: np.ndarray) -> np.ndarray:
    """wheel_torque 模式：腿不用管（PD 到 DEFAULT_Q），轮子动作直接是力矩比例。"""
    act = np.zeros(6)
    act[WHEELS] = np.clip(tau[WHEELS] / TORQUE_SCALE[WHEELS], -1.0, 1.0)
    return act


def torque_to_action(tau: np.ndarray, env, mode: str) -> np.ndarray:
    if mode == "torque":
        return to_action(tau)
    if mode == "wheel_torque":
        return to_action_wt(tau)
    q = env.data.qpos[env.qi]
    dq = env.data.qvel[env.vi]
    return to_action_pv(tau, q, dq)


# ---------------------------------------------------------------- 策略

def make_policy(env, seed: int = 0, device: str = "cpu", arch=(128, 64)):
    from stable_baselines3 import PPO

    return PPO(
        "MlpPolicy", env, n_steps=64, batch_size=64, verbose=0, seed=seed,
        policy_kwargs=dict(net_arch=list(arch)), device=device,
    )


def behavior_clone(model, obs: np.ndarray, act: np.ndarray, epochs: int,
                   batch: int, lr: float, tag: str = "",
                   stand_weight: float = 1.0) -> float:
    import torch

    opt = torch.optim.Adam(model.policy.parameters(), lr=lr)
    device = next(model.policy.parameters()).device
    X = torch.as_tensor(obs, dtype=torch.float32, device=device)
    Y = torch.as_tensor(act, dtype=torch.float32, device=device)

    # 站立样本的动作幅度比运动样本小两个量级，整体 mse 会被运动样本淹没，
    # 于是网络学不会"站立时精确输出小力矩"。这里给站立样本加权。
    weights = np.ones(len(obs), dtype=np.float32)
    if stand_weight != 1.0:
        cmd = obs[:, -19:-16] / np.array([10.0, 0.25, 5.0])
        stand = (np.abs(cmd[:, 0]) < 0.05) & (np.abs(cmd[:, 1]) < 0.05)
        weights[stand] = stand_weight
        print(f"  [{tag}] 站立样本 {stand.mean():.0%}，权重 ×{stand_weight}")
    W = torch.as_tensor(weights, dtype=torch.float32, device=device)

    n = X.shape[0]
    last = float("nan")
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred, _values, _logp = model.policy(X[idx], deterministic=True)
            # W[idx] 是 (batch,)，动作是 (batch, 6)，要扩一维再广播
            loss = (W[idx].unsqueeze(-1) * (pred - Y[idx]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        last = total / n
        if epoch == 0 or epoch == epochs - 1:
            print(f"  [{tag}] epoch {epoch + 1}/{epochs}  mse {last:.6f}")
    return last


# ---------------------------------------------------------------- 采数据

def rollout(env, model, bstate, *, use_teacher: bool, record: bool,
            noise: float = 0.0):
    """跑一个回合。use_teacher=True 时执行 teacher 的动作，否则执行 student 的。

    noise>0 时给 teacher 的动作加高斯扰动（DART）：这样 teacher 会演示
    "被推歪之后怎么救回来"，student 才见过恢复行为——纯专家轨迹里没有这些状态。
    记录的是**干净标签**，不是加噪后的动作。
    """
    mode = env.config.action_mode
    obs, _info = env.reset()
    bstate.clear()
    bstate.update(balance.make_state(env.data, env.base_id))
    obs_list: list[np.ndarray] = []
    act_list: list[np.ndarray] = []
    n = 0
    while True:
        tau = teacher_torque(env, bstate)          # 无论谁在动，每拍都要问 teacher
        label = torque_to_action(tau, env, mode)
        if use_teacher or model is None:
            action = label
            if noise > 0.0:
                action = np.clip(label + np.random.normal(0.0, noise, 6), -1.0, 1.0)
        else:
            action = model.predict(obs, deterministic=True)[0]
        if record:
            obs_list.append(np.asarray(obs, dtype=np.float32))
            act_list.append(np.asarray(label, dtype=np.float32))
        obs, _r, term, trunc, _info = env.step(action)
        n += 1
        if term or trunc:
            break
    return obs_list, act_list, n


# ---------------------------------------------------------------- 主流程

def load_cfg(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_arch(text: str) -> tuple[int, ...]:
    return tuple(int(x) for x in text.replace(" ", "").split(",") if x)


def _roundtrip_error(cfg: dict, steps: int = 300) -> tuple[float, float]:
    """反解自检：decode_action(反解出的 act) 应该还原出教师力矩。

    返回 (腿的 max 误差, 轮子的 max 误差)，单位 N·m。
    """
    from .control import decode_action

    env = build_env(cfg)
    mode = env.config.action_mode
    env.reset()
    bstate = balance.make_state(env.data, env.base_id)
    leg_err = 0.0
    wheel_err = 0.0
    for _ in range(steps):
        tau = teacher_torque(env, bstate)
        act = torque_to_action(tau, env, mode)
        q = env.data.qpos[env.qi]
        dq = env.data.qvel[env.vi]
        _req, applied = decode_action(act, q, dq, env.config.action_mode)
        diff = np.abs(applied - tau)
        leg_err = max(leg_err, float(diff[LEGS].max()))
        wheel_err = max(wheel_err, float(diff[WHEELS].max()))
        _obs, _r, term, trunc, _info = env.step(act)
        if term or trunc:
            env.reset()
            bstate = balance.make_state(env.data, env.base_id)
    env.close()
    return leg_err, wheel_err


def cmd_smoke(args) -> int:
    cfg = load_cfg(args.student_config)
    env = build_env(cfg)
    policy = make_policy(env, device=args.device, arch=parse_arch(args.arch))
    bstate: dict = {}
    obs, act, n = rollout(env, policy, bstate, use_teacher=True, record=True)
    O = np.asarray(obs, dtype=np.float32)
    A = np.asarray(act, dtype=np.float32)
    print(f"action_mode = {env.config.action_mode}")
    print(f"回合步数 {n}")
    print(f"obs {O.shape}  act {A.shape}")
    print(f"动作范围 [{A.min():+.2f}, {A.max():+.2f}]")
    print(f"逐关节均值 {A.mean(axis=0).round(3)}")
    print(f"饱和比例（|a|>0.99）：{(np.abs(A) > 0.99).mean():.1%}")
    if env.config.action_mode == "torque":
        tau = A * TORQUE_SCALE
        print(f"对应力矩（N·m）范围 [{tau.min():+.1f}, {tau.max():+.1f}]")
    elif env.config.action_mode == "wheel_torque":
        print(f"轮子对应力矩（N·m）范围 "
              f"[{(A[:, [2, 5]] * TORQUE_SCALE[[2, 5]]).min():+.1f}, "
              f"{(A[:, [2, 5]] * TORQUE_SCALE[[2, 5]]).max():+.1f}]")
        print("（wheel_torque：腿由 kp=60/kd=2 的 PD 固定，动作只在轮子上）")
    else:
        print("（position_velocity：动作是 PD 目标偏置，腿≈0 说明 teacher 的腿就是 PD 到 DEFAULT_Q）")
    env.close()

    leg_err, wheel_err = _roundtrip_error(cfg)
    print(f"反解自检：腿最大误差 {leg_err:.3f} N·m，轮子最大误差 {wheel_err:.3f} N·m")
    print("（应当 < 0.5 N·m；如果轮子误差随速度变大，说明反解漏了阻尼项）")
    return 0


def cmd_collect(args) -> int:
    cfg = load_cfg(args.student_config)
    env = build_env(cfg)
    policy = make_policy(env, seed=args.seed, device=args.device,
                         arch=parse_arch(args.arch))
    bstate: dict = {}
    t0 = time.time()
    all_obs, all_act = [], []
    for ep in range(args.episodes):
        o, a, n = rollout(env, policy, bstate, use_teacher=True, record=True,
                          noise=args.noise)
        all_obs += o
        all_act += a
        print(f"  episode {ep + 1}/{args.episodes}: {n} 步, 累计 {len(all_obs)} 样本")
    env.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, obs=np.asarray(all_obs, np.float32),
                        act=np.asarray(all_act, np.float32))
    print(f"已保存 {out}  {len(all_obs)} 样本  耗时 {time.time() - t0:.1f}s")
    return 0


def cmd_train(args) -> int:
    data = np.load(args.data)
    obs, act = data["obs"], data["act"]
    cfg = load_cfg(args.student_config)
    env = build_env(cfg)
    policy = make_policy(env, seed=args.seed, device=args.device,
                         arch=parse_arch(args.arch))
    print(f"样本 {len(obs)}，开始行为克隆")
    mse = behavior_clone(policy, obs, act, args.epochs, args.batch, args.lr, tag="bc")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(str(out))
    env.close()
    print(f"已保存 {out}  最终 mse {mse:.6f}")
    return 0


def cmd_dagger(args) -> int:
    cfg = load_cfg(args.student_config)
    env = build_env(cfg)
    policy = make_policy(env, seed=args.seed, device=args.device,
                         arch=parse_arch(args.arch))
    bstate: dict = {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_obs: list = []
    all_act: list = []
    start_it = 0
    if args.resume:
        from stable_baselines3 import PPO

        rdir = Path(args.resume)
        policy = PPO.load(str(rdir / "policy.zip"), env=env, device=args.device)
        d = np.load(rdir / "data.npz")
        all_obs = [row.copy() for row in d["obs"]]
        all_act = [row.copy() for row in d["act"]]
        start_it = args.start_iter
        print(f"[resume] 从 {rdir} 接着跑：已有 {len(all_obs)} 样本，"
              f"迭代号从 {start_it + 1} 开始")
    else:
        # 第 0 轮：teacher 数据（带扰动，让 teacher 演示"歪了怎么救"）
        print(f"[0] teacher 回合（动作噪声 {args.noise}）")
        for ep in range(args.episodes):
            o, a, n = rollout(env, policy, bstate, use_teacher=True, record=True,
                              noise=args.noise)
            all_obs += o
            all_act += a
            print(f"  episode {ep + 1}: {n} 步, 累计 {len(all_obs)}")

    if args.stand_episodes > 0:
        print(f"[stand] 额外采 {args.stand_episodes} 个纯站立回合（补足站立样本）")
        original_get_command = env.get_command
        env.get_command = lambda _t: np.array([0.0, 0.0, 0.25])  # noqa: E731
        try:
            for ep in range(args.stand_episodes):
                o, a, n = rollout(env, policy, bstate, use_teacher=True,
                                  record=True, noise=args.noise)
                all_obs += o
                all_act += a
                print(f"  stand {ep + 1}: {n} 步, 累计 {len(all_obs)}")
        finally:
            env.get_command = original_get_command

    for it in range(start_it + 1, start_it + args.iterations + 1):
        obs = np.asarray(all_obs, np.float32)
        act = np.asarray(all_act, np.float32)
        print(f"[{it}] 行为克隆 {len(obs)} 样本")
        mse = behavior_clone(policy, obs, act, args.epochs, args.batch, args.lr,
                             tag=f"it{it}", stand_weight=args.stand_weight)
        # DAgger：student 自己跑，沿路问 teacher 要标签
        print(f"[{it}] student 回合（沿路采标签）")
        for ep in range(args.episodes):
            o, a, n = rollout(env, policy, bstate, use_teacher=False, record=True)
            all_obs += o
            all_act += a
            print(f"  episode {ep + 1}: {n} 步, 累计 {len(all_obs)}")
        policy.save(str(out_dir / f"policy_it{it}.zip"))
        print(f"  已保存 policy_it{it}.zip  mse {mse:.6f}")

    policy.save(str(out_dir / "policy.zip"))
    np.savez_compressed(out_dir / "data.npz",
                        obs=np.asarray(all_obs, np.float32),
                        act=np.asarray(all_act, np.float32))
    env.close()
    print(f"完成：{out_dir / 'policy.zip'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--student-config", type=Path,
                       default=Path("examples/xingtian_balance_student.json"))
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", default="cpu",
                       help="cpu 或 cuda:0；MuJoCo 仿真始终在 CPU，GPU 只加速小网络")
        p.add_argument("--arch", default="128,64", help="网络宽度，如 256,128")
        p.add_argument("--markov", action="store_true",
                       help="teacher 关掉积分器（无记忆版），行为克隆必须用它")
        p.add_argument("--teacher-kv", default=60.0,
                       help="无记忆模式下 STAND/MOVE 的 k_v（位置积分的替代）")

    p = sub.add_parser("smoke"); common(p); p.set_defaults(fn=cmd_smoke)

    p = sub.add_parser("collect"); common(p)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--noise", type=float, default=0.05, help="teacher 动作上的高斯扰动（DART）")
    p.add_argument("--out", type=Path, default=Path("runs/distill/data.npz"))
    p.set_defaults(fn=cmd_collect)

    p = sub.add_parser("train"); common(p)
    p.add_argument("--data", type=Path, default=Path("runs/distill/data.npz"))
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", type=Path, default=Path("runs/distill/policy.zip"))
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("dagger"); common(p)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--noise", type=float, default=0.05, help="teacher 动作上的高斯扰动（DART）")
    p.add_argument("--resume", type=Path, default=None,
                   help="从已有目录（含 policy.zip 与 data.npz）继续 DAgger")
    p.add_argument("--start-iter", type=int, default=0,
                   help="续训时的迭代编号起点（只影响日志与文件名）")
    p.add_argument("--stand-episodes", type=int, default=0,
                   help="额外采几个纯站立回合（补站立样本，压长时间漂移）")
    p.add_argument("--stand-weight", type=float, default=1.0,
                   help="站立样本在损失里的权重（运动样本动作幅度大两个量级，会淹没站立）")
    p.add_argument("--out", type=Path, default=Path("runs/distill"))
    p.set_defaults(fn=cmd_dagger)

    args = ap.parse_args()
    if getattr(args, "markov", False):
        balance.MARKOV_ONLY = 1.0
        balance.STAND["k_v"] = float(args.teacher_kv)
        balance.MOVE["k_v"] = float(args.teacher_kv)
        print(f"[teacher] 无记忆模式：关掉 psi_ref / s_ref / i_th，"
              f"k_v = {args.teacher_kv}（位置积分用速度增益替代）")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
