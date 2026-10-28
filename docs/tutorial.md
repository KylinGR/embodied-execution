# 刑天 DIABLO GPU 分阶段训练手册

## 1. 项目定位

本项目把公开 CAD 和 Webots 机构恢复为 MuJoCo/MJLab 机器人，在 Linux NVIDIA GPU 上使用 MuJoCo Warp 并行仿真，以 RSL-RL PPO 训练刑天的基础运动能力。

正式训练没有 CPU 训练后备路径。原生 MuJoCo 单环境只负责检查闭环、质量惯量、执行器方向、传感器、重置和故障状态机。

训练课程覆盖：

- 平地原地平衡；
- 前进、后退、加速和减速；
- 原地左转、右转和行驶转向；
- 一个回合内多次启动和停止；
- 35° 上坡、坡顶和受控下降；
- 连续变化崎岖地形上的综合运动；
- 轮子碾压 2–5 cm 木板、圆柱和胶囊障碍；
- 融合平地、崎岖地形、坡面和障碍的长时综合任务。

仓库不附带成熟策略。只有目标服务器执行完整训练，并通过 validation、盲测和边界测试后，才能声明相应能力已经获得。

## 2. 硬件和模型参数

公开产品页给出的刑天参数包括：6 个直驱关节、整机 22.9 kg、最高持续速度 2 m/s、站立负载 4 kg、匍匐负载 80 kg。项目保留 CAD 的六主动关节、四被动关节和两个平行四边形闭环，并把 CAD 各刚体质量与惯量按同一比例归一化到 22.9 kg。

课程文件中另有两个基座高度：

- `0.25175 m` 与当前默认姿态和轮底几何不一致，已作为旧值替换；
- `0.270943 m` 等于当前 CAD 默认姿态的轮底高度再加 5 mm，作为实际训练目标。

`0.37958 m` 保留为课程整机高度包络。官网当前页面给出的 491 mm 属于另一版整机尺寸，未用于覆盖本仓库的 CAD 模型。

关键模型参数：

| 参数 | 值 |
|---|---:|
| 主动电机 | 6 |
| 被动关节 | 4 |
| 平行四边形闭环 | 2 |
| 总质量 | 22.9 kg |
| 主轮半径 | 0.1 m |
| 堵转力矩 | 17 Nm |
| 额定力矩 | 9.6 Nm |
| 空载转速 | 23.038 rad/s |
| 物理步长 | 0.002 s |
| 控制周期 | 0.01 s |

## 3. 训练软件栈

| 组件 | 版本/作用 |
|---|---|
| MJLab | 1.6.0，管理器式环境和 RSL-RL |
| MuJoCo Warp | 3.11，GPU 批量物理 |
| MuJoCo | 3.11，CPU 校核和模型编译 |
| RSL-RL | 由 MJLab 锁定，PPO |
| PyTorch | CUDA 策略网络与张量计算 |

要求 Linux x86_64、Python 3.11/3.12、NVIDIA GPU，以及能运行所选 PyTorch CUDA wheel 的驱动。安装脚本默认使用 cu128 wheel；不匹配时可通过 `TORCH_INDEX_URL` 指定官方 PyTorch CUDA 索引。`gpu-doctor` 会在 CUDA 不可用时返回错误，不会降级到 CPU 训练。

## 4. 安装

```bash
git clone https://github.com/KylinGR/embodied-execution.git
cd embodied-execution
bash scripts/setup.sh
source .venv/bin/activate
```

安装脚本执行以下检查：

```bash
diablo fetch
diablo doctor
diablo gpu-doctor
```

`diablo fetch` 从固定 Git 提交获取 CAD、网格和官方 Webots 机构，不对上游文件做隐式重置。若已有目录版本不一致，应保留现有修改并使用新的资源目录。

## 5. 观测接口

Actor 输入固定为 `[B,160]`：

| 部分 | 维度 | 内容 |
|---|---:|---|
| 本体历史 | 125 | 当前 25 维本体观测的五帧历史 |
| 地形扫描 | 35 | 前向 `7×5` 高度网格 |

单帧本体观测 25 维：

- 机体角速度 3；
- 机体系投影重力 3；
- 指令 `[前进速度, 偏航角速度, 基座高度]` 3；
- 髋膝相对默认位置 4；
- 六电机速度 6，轮速度按前进方向统一符号；
- 上一时刻动作 6。

Critic 使用相同本体与地形输入，并增加真实基体速度、基座高度、地形表面法向和轮地接触。该特权信息不进入 Actor，也不进入 ONNX。

## 6. 动作接口

动作顺序固定为：

```text
left_hip, left_knee, left_wheel,
right_hip, right_knee, right_wheel
```

髋膝动作范围是默认姿态的 `±0.5 rad`。轮动作目标速度由阶段控制参数决定，正动作沿机体 `+x` 前进。轮执行器在 GPU 上执行：

- PI 目标轮速跟踪；
- 积分限幅；
- 轮速目标斜率限制；
- 轮速前馈；
- 17 Nm 堵转和随转速变化的可用力矩包络。

髋膝使用 MJLab 原生位置执行器，不把轮位置误当作轮速。

## 7. 课程顺序

| 顺序 | 阶段 | budget | 重点 |
|---:|---|---:|---|
| 1 | `flat_balance` | 10M | 高度、直立、静止和漂移 |
| 2–11 | `flat_motion_020` … `flat_motion_200` | 每阶段 8M | 0.2–2.0 m/s 速度阶梯 |
| 12 | `flat_motion` | 12M | 平地运动综合 |
| 13 | `flat_turning` | 10M | 原地和行驶转向 |
| 14 | `flat_start_stop` | 10M | 三次以上启停 |
| 15 | `slope` | 14M | 35° 坡道 |
| 16 | `rough` | 16M | 连续崎岖地形 |
| 17 | `roller_obstacle` | 10M | 低高度可碾压障碍 |
| 18 | `composite` | 22M | 长时综合任务 |

budget 是新增环境 transition 总数。默认每环境 rollout 为 24 步，迭代数为：

```text
ceil(budget / (num_envs × 24))
```

各阶段 episode 时长取该阶段最长的 profile。每个环境按编号轮流分配 profile，而不是随机丢 profile，从而保证 GPU batch 内覆盖全部运动模式。

## 8. 地形

- 平地阶段使用无限平面。
- 坡面地形包含起步平路、35° 上升段、坡顶和下降段。
- 原始坡面 JSON 的 5 m 起步距离和 8 m 坡长无法被其 25 s、约 3.58 m 的前进 profile 走到，因此坡面专项把上升段压缩到 3.5 m；35° 角度和 3 m 最低前进 gate 保持不变。
- 崎岖地形使用 0.01–0.04 m 连续高度场，训练难度按地形行递增，单点局部坡度限制在 35° 内。
- 障碍地形使用 2–5 cm 高、横跨两条轮迹的 box、cylinder 和 capsule；前向与反向任务都统计碾压通过次数。
- 综合任务依次组合平地、崎岖接近段、35° 坡、坡顶、下降、障碍和终点。地形长度扩展到 60 m，以容纳课程 profile 约 56.65 m 的理论前进距离。

地形与障碍使用独立碰撞层：机器人和地形可接触，机器人自身碰撞被屏蔽。

## 9. 奖励设计

奖励权重直接来自 `diablo_training/curriculum.json` 和 `flat_motion_speed_stages.json`，阶段 JSON 的 `reward_defaults` 与 `overrides` 优先于公共默认值。

正向形状项：

- 前进速度、低于目标速度、偏航速度和基座高度；
- 直立、俯仰、静止、保持漂移；
- 加速度、制动、原地转向；
- 轮速与机体速度匹配；

惩罚项：

- 横向速度、动作变化率、关节力矩和超过额定力矩；
- 关节软限位、轮滑、后滑；
- 左右轮中心偏差和直行轮速差；
- 垂直速度、横滚/俯仰角速度；
- 非回馈机械功率和跌倒。

事件奖励来自阶段 JSON：

- `obstacle_pass`；
- `slope_crest`；
- `finish`。

每回合同时累计速度/偏航 MAE、直立比例、俯仰和横滚、高度误差、轮滑、后滑、轮同步、停车速度、保持漂移、前进距离、启停次数、障碍数和终点状态。

## 10. 训练

查看课程和默认迭代数：

```bash
diablo list-stages --n-envs 4096
```

一次执行完整课程：

```bash
diablo course-train \
  --n-envs 4096 \
  --gpu-ids 0 \
  --log-root logs/rsl_rl
```

每个前置阶段结束时会评估最后 checkpoint；最终综合阶段改为检查该 run 的全部周期 checkpoint。初段验证种子为：

```text
2001, 2002, 2003
```

每个 seed、每个 profile 都必须通过全部 gate，才进入下一阶段。失败时会停止，不自动带着不达标策略继续训练后续复杂地形。

单阶段运行：

```bash
diablo train --stage flat_turning --n-envs 4096 --gpu-ids 0
```

续训必须指定 run 下的 checkpoint：

```bash
diablo train \
  --stage flat_motion_020 \
  --resume logs/rsl_rl/diablo_xingtian_course/<run>/model_<iter>.pt
```

多 GPU：

```bash
diablo course-train --n-envs 8192 --gpu-ids 0,1
```

## 11. policy-best 两级筛选

只有最终 `composite` run 参与 policy-best 筛选。该 run 每 10 次迭代保存一个 checkpoint。

初筛：

```text
2001, 2002, 2003
```

任一 seed、任一 profile 的任一 gate 失败即淘汰。初筛通过后，使用完全不重叠的五个验证 seed：

```text
2004, 2005, 2006, 2007, 2008
```

五个 seed 必须全部通过全部 gate。其后再对通过者计算综合得分：

| 类别 | 权重 |
|---|---:|
| 完成度 | 25% |
| 速度和转向 | 25% |
| 姿态和高度 | 25% |
| 坡面和障碍 | 15% |
| 轮滑和机械效率 | 10% |

最大阈值指标使用 `1 - observed/limit`，最小比例指标在 `[min,1]` 中归一化，进度和障碍数相对最小值归一化。最终按平均 gate 分数排序；并列时依次比较最差 seed 分数、最小 gate 余量、平均速度 MAE、平均偏航 MAE，最后选择迭代更早的 checkpoint。

单独筛选：

```bash
diablo select-best \
  --run-dir logs/rsl_rl/diablo_xingtian_course/<run> \
  --device cuda:0
```

生成：

```text
policy-best.pt
policy-best.onnx
selection.json
```

无 checkpoint 完全通过时，命令返回非零状态，不会把最后一个 checkpoint 冒充 best。

选择完成后再执行盲测和边界测试：

```text
blind: 3101–3110
limit: 4101–4103
```

这两组结果仅用于最终泛化报告，不参与 policy-best 选择。

如果 blind 或 limit 中任一 seed/profile 未通过 gate，`select-best` 仍保留已选权重和报告，但返回退出码 2，表示最终泛化验收未通过。

## 12. 回放、评估和导出

查看 MJLab 回放：

```bash
play Mjlab-Xingtian-Composite \
  --checkpoint-file <run>/policy-best.pt \
  --device cuda:0
```

执行 CLI gate 评估：

```bash
diablo evaluate \
  --stage composite \
  --checkpoint <run>/policy-best.pt \
  --seeds 2001,2002,2003
```

导出任意 checkpoint：

```bash
diablo export \
  --stage composite \
  --checkpoint <run>/policy-best.pt \
  --output runs/final-export
```

ONNX 的 Actor 输入为 `[N,160]`，输出为 `[N,6]`。导出成功只证明网络与接口一致；实机部署仍需零位、电机驱动、通信延迟、负载和安全性标定。

## 13. 测试

无 GPU 时仍可运行：

```bash
bash scripts/check_training.sh
```

该检查包括 Python 语法、课程顺序、seed 互斥、gate 映射、评分和 checkpoint 选择逻辑。安装了 MuJoCo/Gymnasium 时还会运行 CPU 机构、闭环、执行器方向、传感器和故障测试。

在 CUDA Linux 服务器上：

```bash
DIABLO_GPU_CHECK=1 bash scripts/check_training.sh
```

额外运行 GPU doctor、随机动作 smoke、短 PPO、checkpoint 重载、单回合评估和 ONNX 导出。

完整课程包含约 184M transition。CI 中不执行完整训练，也不把短训练结果作为运动能力证据。

### 交付检查记录

2026-10-28 在 Windows 源码工作区完成：

- `python -m compileall` 通过；
- 14 项课程、gate、评分和 CPU 模型测试全部通过；
- 使用 MuJoCo 3.11.0 和锁定公开资源验证 22.9 kg、六执行器、两个闭环、正向轮动作、FK、虚功和重置；
- wheel 构建成功，包含两份 JSON 和 `mjlab.tasks` entry point；
- 当前机器无 NVIDIA GPU、MuJoCo Warp 和 MJLab，因此未声称 GPU smoke 或策略训练通过。

## 14. 常见问题

| 现象 | 处理 |
|---|---|
| `CUDA is unavailable` | 检查驱动、CUDA 可见设备；项目不会使用 CPU 训练 |
| `nvidia-smi` 不存在 | 安装 NVIDIA 驱动，不要只安装 CUDA toolkit |
| 显存不足 | 降低 `--n-envs`；迭代数会自动按 transition budget 重新计算 |
| `Model source is missing` | 执行 `diablo fetch` |
| 上游资源版本不一致 | 保留现有目录，另用干净资源目录重新 fetch |
| checkpoint shape 不匹配 | 使用同一模型、观测和动作契约；不要强载旧训练权重 |
| 某阶段 gate 失败 | 保留该 run，查看 `result.json` 的逐 seed/profile 指标；不要跳过阶段 |
| `No candidate checkpoint passed` | 增加训练预算或调参后生成新 run，系统不会伪造 best |
| EGL/渲染错误 | 训练和 gate 评估不需要 viewer；先不启用视频 |

## 15. 资料与边界

- 模型资源和许可证边界见 [resources](resources.md)。
- 观测、动作、执行器和时钟细节见 [model-interface](model-interface.md)。
- 当前仓库没有实机授权、标定电机曲线、热模型或真实负载验证。
- 所有训练与评估结果均属于仿真范围；存在 sim-to-real gap。
