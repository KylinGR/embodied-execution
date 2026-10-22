# A 阶段报告：奖励修复与对照实验

> 日期：2026-10-21 ~ 10-22
> 分支：`base/yue/reward-fix`
> 任务：`examples/xingtian_stage_a.json`（平地：平衡 / 前进 / 后退 / 转向 / 启停）
> 评测：`xingtian_eval` + `select-best`，独立种子 `2002`

---

## 0. 结论

| 项 | 状态 |
|---|---|
| 五阶段存活 | ✅ 5/5，全部跑满时长 |
| 姿态 | ✅ `min_upright` 全部 ≥ 0.965 |
| 前进 / 后退 / 启停 | ✅ `yaw_mae` 0.078 / 0.112 / 0.082 |
| 直线行驶 | ✅ 视频确认，无侧翻、无单向漂移 |
| **转向精度** | ⚠️ `turn` `yaw_mae` **0.693**（指令 0.8 rad/s，实际约 0.08） |

**交付权重**：`runs/A-rewardfix3-2m/selection/policy-best.zip`（1,900,000 步检查点）
服务器备份：`~/outputs/A-stage/`

---

## 1. 改了什么

只动 `diablo_training/xingtian.py` 里 `LocomotionReward` 的奖励项，**没碰观测、动作、物理**。

```python
# 改前
"yaw_tracking": w.yaw_tracking * np.exp(-(yaw_error**2) / 0.09),
"height": w.height * np.exp(-(height_error**2) / 0.0004),

# 改后
"yaw_tracking": w.yaw_tracking * (
    np.exp(-(yaw_error**2) / 0.25) if abs(command[1]) > 0.1
    else 0.5 * np.exp(-(gyro[2]**2) / 0.02)
),
"height": w.height * np.exp(-(height_error**2) / 0.005),
```

三处变化：

| 项 | 前 | 后 | 理由 |
|---|---|---|---|
| `yaw_tracking` 跟踪核 | `/ 0.09`（σ=0.21） | `/ 0.25`（σ=0.35） | 原核过窄，误差 0.92 落在 4.4σ 外，`exp(−9.5)≈8e−5`，梯度消失 |
| `yaw_tracking` 守卫 | 无 | 无转向指令时切换到"抑制偏航"分支 | **消除"不转就得分"的漏洞** |
| `height` 核 | `/ 0.0004`（σ=**1.4 cm**） | `/ 0.005`（σ=5 cm） | σ 只有 1.4 cm，该项从训练开始到结束恒为 0 |

**接口不变**：观测 `[125]`（25 维 × 5 帧）、动作 `[6]`。

---

## 2. 为什么改：根因分析

### 2.1 现象

复评原始 A 阶段权重（`runs/xingtian-flat-a1`，200 万步）：

```
turn   duration 12.0 s（满）  yaw_mae 0.923  指令 0.8 rad/s
```

误差**比指令本身还大**。抽帧确认：机器人确实在转，但方向与指令相反，9.6 秒累积约 66°。

### 2.2 三个跟踪项的 σ

| 项 | `a` | σ | 实测误差 | 几倍 σ | 该项奖励 |
|---|---|---|---|---|---|
| `forward_tracking` | 0.04 | 0.141 m/s | 0.157 | **1.1σ** | **0.54** ✅ |
| `yaw_tracking` | 0.09 | 0.212 rad/s | 0.923 | **4.4σ** | **8e-5** ❌ |
| `height` | 0.0004 | **0.014 m** | ~0.055 | **3.9σ** | **5e-4** ❌ |

**误差落在 4σ 外，梯度基本为零。**

### 2.3 逆向激励

命令调度器分布：20% 零速 / 25% 前进 / 20% 后退 / 25% 转向 / 10% 混合
→ **75% 的时间里转向指令是 0**。

| 情形 | 占比 | 不转时 `yaw_tracking` | 努力转时 |
|---|---|---|---|
| 无转向指令 | **75%** | **0.77** | ≈ 0（把直线走歪） |
| 有转向指令 | 25% | ≈ 0 | ≈ 0（4σ 外，够不着） |

**期望收益：不转 ≈ 0.58，转 ≈ 0。**

→ 策略不是在"学不会"，而是在**正确地最大化一个设计有缺陷的奖励**。

### 2.4 账本验证

`forward` 段每步：`forward 0.648 + yaw 0.772 + upright 0.656 − 惩罚 ≈ 2.0`
× `dt 0.01` × 3000 步 ≈ **60**，实测回合奖励 **43–58**。账对得上。

---

## 3. 对照实验：守卫强度

三层结构，同预算（200 万步）、同配置、同种子（2002），只改守卫强度。

### 3.1 关键数据

| 守卫 | 聚合 `yaw` | 聚合 `speed` | `forward` `up` | `forward` `yaw` | `backward` `yaw` |
|---|---|---|---|---|---|
| `0.3 × / 0.04` | 0.458 | 0.203 | 0.957 | 0.113 | — |
| `1.0 × / 0.01` | 0.263 | 0.124 | **0.705** ❌ | 0.254 | 0.111 |
| **`0.5 × / 0.02`** | **0.214** | **0.092** | **0.975** | **0.078** | 0.112 |

### 3.2 三轮各自的现象

**① `0.3 × / 0.04`（太弱）**

转 0.113 rad/s 只损失 **0.066 分**，而 `forward_tracking` 是 1.2 分——阻尼弱了 18 倍。

结果：`forward` 段 12 秒内转了约 **78°**，走成弧线，机体前向速度被削 → `speed_mae 0.469`。

**② `1.0 × / 0.01`（太强）**

转 0.113 损失 **0.58 分**，"不转"的收益过大，策略开始**用腿部硬压偏航**，把机身拧翻。

结果：`forward` 的 `min_upright` 崩到 **0.705**（倾斜约 45°），四帧视频显示一路侧翻。

**③ `0.5 × / 0.02`（中间档）✅**

姿态回到 0.975，航向 0.078，两个目标同时满足。

> **结论：权重不是越大越好。过了某个点，它会逼策略去用错误的手段（腿部）完成本应由轮子差速完成的动作。**

---

## 4. 另一个发现：吞吐与资源强相关

| 时段 | 吞吐 | 机器状态 |
|---|---|---|
| 白天 | 543 t/s | 其他任务占满 6 张卡 |
| 白天 | 665 / 527 t/s | 部分占用 |
| **夜间/独占** | **1369 t/s** | 机器全空 |

**同样的代码和配置，独占时快 2.1 倍。**

时间分配：MuJoCo 采集占 16%，PPO 更新占 84%。

`nvidia-smi` 显示 GPU 利用率仅 **6%**、显存 347 MiB —— 模型只有 MLP `[128,64]`（约 2.5 万参数），
每轮仅 1024 个样本，GPU 大量时间在等而不是算。

> **待验证**：`--device cpu` 是否更快。WSL 上纯 CPU 跑旧配置是 1446 t/s。

---

## 5. 已知限制

### 5.1 转向精度不足（未解决）

`turn` 的 `yaw_mae 0.693`，从 `turn.npz` 里读 `gyro_z` 序列确认：**指令阶跃到 0.8 时毫无响应**，
全程维持在 0.1 左右。

**已排除**：

- 不是测量错误——帧里约 40° 的转角与 0.08 rad/s × 8.6 s 吻合
- 不是物理能力——瞬时能冲到 1.16 rad/s

**已试过并证伪的方案**：把 `max_yaw_rate` 全局从 1.5 降到 0.5。

| 指标 | 1.5 | 0.5 |
|---|---|---|
| `turn yaw` | 0.693 | **0.502** ✅ |
| **`backward yaw`** | 0.111 | **0.739** ❌ |
| `forward yaw` | 0.078 | 0.236 ❌ |

原因：`CommandScheduler` 里

```python
# turn 分支
yaw_rate = sign * uniform(0.25, 1.0) * max_yaw_rate
# 零速 / 前进 / 后退分支
yaw_rate = uniform(-0.2, 0.2) * max_yaw_rate
```

**全局参数同时缩放了"转向幅度"和"直行时的偏航扰动"。** 降下来虽然让 turn 改善，
但直行时遇到的扰变小了，航向保持的训练强度跟着掉。

> **正确修法**：在 `CommandScheduler` 的 turn 分支**单独**限制振幅，不动全局参数。
>
> 另一个方向：用"达成比例"（`min(1, |actual|/|cmd|)`）替代高斯核，让"转不到"不给分、
> "转到一半"给一半分，梯度恒定不消失。

### 5.2 体高指令疑似不可达

指令 `height = 0.25 m`，实测站立时 `height_mean` 稳定在 **0.1945 m**，差 5.5 cm。

而 `XingtianConfig` 的 `min_height / max_height = 0.16 / 0.34` —— **范围与实测严重不匹配**。

`height_error = base_z − ground_z − command[2]` 的口径需要确认：
这个范围是按"基座 z 坐标"定的，还是按"腿长"定的？

**影响所有阶段**，不只是 A。

### 5.3 `std` 持续上涨

`std`：1.0 → 1.1 → 1.38 → 1.49 → **1.96**

`ent_coef = 0.008` 的熵奖励可能偏高，策略在往"高熵、乱试"的方向漂。
`clip_fraction` 也从 0.136 涨到 0.223（22% 样本撞到裁剪边界），更新幅度过大。

**候选改法**：`ent_coef 0.008 → 0.002`。

---

## 6. 复现

```bash
conda activate /home/yuewenzhe/envs/diablo
cd ~/projects/embodied-execution
```

**训练**

```bash
CUDA_VISIBLE_DEVICES=1 diablo train --config examples/xingtian_stage_a.json \
  --device cuda:0 --vec-env subproc --n-envs 16 --cpu-threads 1 --gpu-fraction 0.5 \
  --steps 2000000 --output runs/A-rewardfix3-2m
```

**选最优检查点**（终点通常不是最好的）

```bash
diablo select-best --run runs/A-rewardfix3-2m \
  --stages stand forward backward turn start_stop --seed 2002
```

**评测**

```bash
python -m diablo_training.xingtian_eval \
  --model runs/A-rewardfix3-2m/selection/policy-best.zip \
  --stages stand forward backward turn start_stop --seed 2002 \
  --output runs/eval-r3
```

**录像 + 抽帧**

```bash
MUJOCO_GL=egl python -m diablo_training.xingtian_eval \
  --model runs/A-rewardfix3-2m/selection/policy-best.zip \
  --stages forward turn --seed 2002 --video --output runs/eval-r3-video
```

```bash
python -c "import imageio.v2 as iio;r=iio.get_reader('runs/eval-r3-video/rollout.mp4');[iio.imwrite(f'runs/f{i:03d}.png', r.get_data(i)) for i in (0,80,160,240)]"
```

---

## 7. 接手建议

**优先级由高到低：**

1. **先修 `height` 的口径** —— 它影响所有阶段，而且是纯配置问题，改起来最快
2. **再攻 `turn`** —— 按 5.1 的两条路（单独限振幅 / 换成达成比例）
3. **然后才考虑 `ent_coef`** —— `std` 涨到 1.96 已经不正常，但它可能是前两个问题的结果而非原因
4. B 阶段（`xingtian_stage_b_l1.json` + `--warm-start`）

**几条操作纪律：**

- **一律用 `selection/policy-best.zip`**，不要用 `policy.zip`（三轮实验里，终点都不是最优的）
- 改奖励**一次只改一处**，留对照组
- **哈希、版本号让机器比**（`print(a == b)`），别看截图
- 训练前先 `nvidia-smi`，独占和共享差 2 倍速度

---

## 8. 环境备忘

| 项 | 值 |
|---|---|
| 服务器 | `3090-6.grifcc.top`，Ubuntu 24.04，6×RTX 3090，**无外网**，无 sudo |
| 环境 | `conda activate /home/yuewenzhe/envs/diablo` |
| Python | 3.12 · torch 2.14.1+cu130 · mujoco 3.8.1 |
| 代码 | `~/projects/embodied-execution` |
| 依赖安装 | `pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple` |
| 并行度 | `--n-envs 16`（32 只快 8.9%，不值） |
