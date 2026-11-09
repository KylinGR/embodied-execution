# 刑天 DIABLO GPU 分阶段训练手册

## 1. 项目定位

本项目把公开 CAD 和 Webots 机构恢复为 MuJoCo/MJLab 机器人，在 Linux NVIDIA GPU 上使用 MuJoCo Warp 并行仿真，以 RSL-RL PPO 训练刑天的基础运动能力。

正式训练没有 CPU 训练后备路径。原生 MuJoCo 单环境只负责检查闭环、质量惯量、执行器方向、传感器、重置和故障状态机。

训练课程覆盖：

- 平地原地平衡；
- 前进、后退、加速和减速；
- 原地左转、右转和行驶转向；
- 一个回合内多次启动和停止；
- 30° 上坡、坡顶和受控下降；
- 连续变化崎岖地形上的综合运动；
- 轮子碾压 2–5 cm 木板、圆柱和胶囊障碍；
- 融合平地、崎岖地形、坡面和障碍的长时综合任务。

仓库不附带成熟策略。只有目标服务器执行完整训练，并通过 validation、盲测和边界测试后，才能声明相应能力已经获得。

## 2. 硬件和模型参数

公开产品页给出的刑天参数包括：6 个直驱关节、整机 22.9 kg、最高持续速度 2 m/s、站立负载 4 kg、匍匐负载 80 kg。产品页质量是外部产品规格，不是模型输入。机器人模型严格采用原始主分支的 CAD 质量、惯量和质心分布，编译质量为 `21.019506 kg`；不做质量缩放，也不覆盖 CAD 质心。

恢复正常默认关节姿态后：

- 默认健康关节角为 `[0.969324,-1.938616,0,0.969324,-1.938616,0] rad`；
- 默认健康站姿下，轮底接地时基座原点为 `z=0.350 m`，整机外形高约 `0.477859 m`；
- 复位从基座 `z=0.355 m` 开始，保留约 `5 mm` 离地间隙；
- 产品资料中的上部高度包络 `0.491 m` 不再作为训练高度指令。

官网整机质量、整机高度和轮径中与公开 CAD 不一致的数值只作外部参考，不用于覆盖锁定资源。

关键模型参数：

| 参数 | 值 |
|---|---:|
| 主动电机 | 6 |
| 被动关节 | 4 |
| 平行四边形闭环 | 2 |
| CAD 总质量 | 21.019506 kg |
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

训练命令的 `--gpu-ids` 使用物理 GPU 编号。程序会在导入 CUDA 前设置 `CUDA_VISIBLE_DEVICES`，并把子进程参数重映射为本地编号；例如 `--gpu-ids 2` 在训练子进程中对应 `cuda:0`。遮蔽之后的 `--device` 也使用本地编号。

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

Actor 输入固定为 `float32[B,125]`，即当前 25 维本体观测的五帧历史。Actor 不包含地形扫描，输入输出维度和类型不得随模型物理资料恢复而改变。

单帧本体观测 25 维：

- 机体角速度 3；
- 机体系投影重力 3；
- 指令 `[前进速度, 偏航角速度, 航向误差]` 3；
- 髋膝相对默认位置 4；
- 六电机速度 6，轮速度按前进方向统一符号；
- 上一时刻动作 6。

Critic 使用相同本体历史，并增加真实基体速度、基座高度、横向位置误差、世界横向速度、目标路径法向速度、`7×5` 地形扫描、地形表面法向和轮地接触。该特权信息不进入 Actor，也不进入 ONNX。

Actor 不保存外部循环状态。五个 25 维帧在模型内部分别编码，随后经过核长 3、扩张率 1/2 的因果残差卷积；最新帧、时间均值和指令编码融合后进入 MLP 主干。髋膝使用 4 维输出头，轮速使用独立 2 维残差输出头。Critic 使用独立时序编码器，并分别处理 `7×5` 高度扫描和 11 维特权信息。该结构不增加观测、动作或 ONNX 输入输出。

## 6. 动作接口

动作顺序固定为：

```text
left_hip, left_knee, left_wheel,
right_hip, right_knee, right_wheel
```

髋膝动作范围是默认姿态的 `±0.5 rad`。轮动作是解析轮速目标的残差，正动作沿机体 `+x` 前进。轮速目标由前进和偏航指令计算左右轮名义速度，再叠加最多 `4 rad/s` 的网络残差，并限制在电机空载转速以内。轮执行器在 GPU 上执行：

- PI 目标轮速跟踪；
- 积分限幅；
- 轮速目标斜率限制；
- 轮速前馈；
- 17 Nm 堵转和随转速变化的可用力矩包络。

髋膝使用 MJLab 原生位置执行器，不把轮位置误当作轮速。

## 7. 课程顺序

| 顺序 | 阶段 | budget | 重点 |
|---:|---|---:|---|
| 1 | `flat_balance` | 40M | 高度、直立、静止和漂移 |
| 2 | `flat_motion_020` | 8M | 0.2 m/s 最低直行速度 |
| 3 | `flat_motion_030` | 6M | 0.3 m/s |
| 4–7 | `flat_motion_040` … `flat_motion_100` | 每阶段 8M | 0.4–1.0 m/s |
| 8 | `flat_motion` | 12M | 平地运动综合 |
| 9 | `flat_turning` | 10M | 原地和行驶转向 |
| 10 | `flat_start_stop` | 10M | 三次以上启停 |
| 11 | `slope` | 14M | 30° 坡道 |
| 12 | `rough` | 16M | 连续崎岖地形 |
| 13 | `roller_obstacle` | 10M | 低高度可碾压障碍 |
| 14 | `composite` | 22M | 长时综合任务 |

独立转向专项课程从已通过 080 门限的 actor 热启动。提高 `target.straight_speed_max_mps` 后，主课程会自动纳入对应的速度阶梯阶段。

| 顺序 | 阶段 | 速度指令 | 偏航角速度 | budget |
|---:|---|---:|---:|---:|
| 1 | `flat_turn_in_place_040` | 0 | ±0.4 rad/s | 4M |
| 2 | `flat_turn_moving_040` | ±0.4 m/s | ±0.4 rad/s | 6M |
| 3 | `flat_turn_in_place_080` | 0 | ±0.8 rad/s | 6M |
| 4 | `flat_turn_moving_080` | ±0.6 m/s | ±0.8 rad/s | 8M |
| 5 | `flat_turn_in_place_120` | 0 | ±1.2 rad/s | 8M |
| 6 | `flat_turn_moving_120` | ±0.9 m/s | ±1.2 rad/s | 10M |

`flat_turn_moving_080` 是保留的历史任务标识；720 实际训练和 rollout 使用的是 `0.6 m/s × 0.8 rad/s`。
所有新训练阶段统一使用默认姿态实测的轮中心半轮距 `0.24125 m`。策略内部 yaw 估计、轮速前馈和奖励归一化使用同一值；转向阶段启用 common residual 限幅 `0.2`。
| 7 | `flat_turn_in_place_160_200` | 0 | ±1.6–2.0 rad/s | 12M |

原地 0.4 rad/s 阶段分别执行顺时针和逆时针一整圈。从 0.8 rad/s 开始的原地阶段，两个 profile 分别为顺时针和逆时针旋转：静止 2 秒、1 秒建立角速度、保持目标角速度 10 秒、1 秒回零、静止 2 秒。

行驶 S 弯阶段的前进和后退 profile 结构相同：静止后加速到速度上限，保持直线行驶，再分别经历顺时针、逆时针角速度斜坡和保持段；最终回正、减速到零。`flat_turn_moving_120` 使用 `±0.9 m/s` 和 `±1.2 rad/s`，命令曲率半径为 `0.75 m`。

转向阶段的终止倾角不随角速度放宽：低角速度使用 18°，1.2 rad/s 阶段收紧到 16°。正常转弯所需的滚转由 roll 奖励、`max_roll_deg` gate 和 TensorBoard 中独立的 roll/pitch 指标约束。

每个阶段还记录实际 `yaw_rate_peak_rad_s`。所有行驶转弯阶段不超过 1.2 rad/s；原地最终边界阶段实测峰值不得超过 2.0 rad/s。

所有转向阶段同时使用 yaw 欠速 Huber 惩罚，避免策略停在高斯奖励的零梯度区。所有阶段的训练容差统一收紧到对应 gate 的 80% 以内；例如原地 0.4 rad/s 阶段的 `max_yaw_mae_rad_s` 为 `0.06`，奖励容差为 `0.048 rad/s`。

yaw 目标在所有阶段都增加闭环航向修正：在原始 yaw 指令上叠加 `-0.75 × heading_error`，修正量限制为 `±0.2 rad/s` 并最终钳位到硬限幅 `±2.0 rad/s`。该目标同时用于策略观测、轮速前馈、奖励和 yaw 指标，因此终点静止阶段可以主动补转航向，而不会因“零 yaw 命令”抑制纠偏。

原地阶段训练包含至少 25% 的前进/后退排练。`flat_turn_in_place_120` 提高到 30%，并使用 `±0.8 m/s` 排练以保留已训练的直行能力；其余原地阶段使用 `±0.4 m/s`。排练使用 2 秒静止、4 秒加速、4 秒巡航、4 秒减速和 2 秒静止；`flat_turn_in_place_040` 的总时长更短。排练使用独立 `rehearsal_gate` 评估，结果写入 `result.json` 的 `rehearsal` 字段，不改变原转向 gate 或主 `passed` 结果。

奖励与 gate 还同时累计带符号的实际转角和指令转角，记录 `actual_turn_angle_rad`、`commanded_turn_angle_rad`、`turn_angle_error_rad`，使用 `max_turn_angle_error_rad` 门限。所有转向阶段对轮中心偏移增加小权重惩罚，而不只在直行阶段优化轮对齐。

转向时对轮中心偏移的惩罚超过训练容差后继续连续增长，不再在 gate 附近饱和。训练和回放同时记录带符号轮心偏移、左右髋膝相对默认姿态的偏移，以及左右髋/膝差值；这些指标按 clockwise/counterclockwise profile 分开记录，用于定位方向不对称。`leg_symmetry` 奖励只在转向控制期间启用。

转向课程恢复 RSL-RL 自适应学习率，并在转向 runner 内把学习率上限硬限制为 `1e-3`，与同类轮足/足式项目常用的初始学习率对齐，同时阻止原生调度把学习率继续推到 `1e-2`。actor 的高斯 std 不再在训练期间做持续范围钳位；warm-start 默认保留源 actor 的探索参数，使用 `--initial-std` 可显式覆盖。熵系数保持 `0.002`，第一轮先验证动作饱和与 KL 调度，避免同时改变多个变量。

轮速动作入口对偏航命令实施 `±2.0 rad/s` 硬钳位，避免部署包装器绕过课程上限。

轮速残差分为共模 `(left+right)/2` 与差模 `left-right`。只有有效 yaw 指令超过 `0.05 rad/s` 时才把共模限制在归一化动作的 `±0.2`，直行阶段保留原残差；差模不限幅。`wheel_common_residual` 奖励同样只在转向阶段生效。共模/差模还会记录到 TensorBoard、evaluation 和回放 CSV。

自适应速度课程由 `flat_adaptive_speed_stages.json` 定义，所有可调上限和时间来自 `training_objectives.json`。保护网格从平衡、0.2–0.8 m/s 直行、`±0.4/±0.8 rad/s` 原地转向和已验证的 `0.6 m/s × 0.8 rad/s` 行驶转弯开始，按每个二维单元的核心健康成功率扩张；最终再用任务 gate 验收。道路转弯通过 `R=|vx|/|wz|` 校验，仅激活 `0.667–10 m` 内的单元；行驶转弯速度单独限制为 `0.9 m/s`。

保护集以 720 的实际训练上限为准：平衡、直行 0.2–0.8 m/s、原地 0.4–0.8 rad/s，以及 `0.4 × 0.4`、`0.6 × 0.8` 两个行驶转弯单元。原地 1.0/1.2 rad/s、直行 1.0 m/s 和 `0.8 × 0.8` 都属于后续扩张目标。课程健康与扩张只检查持续时间、直立、俯仰、横滚和高度安全：protected health 要求连续两个窗口达到 0.8，普通单元连续两个窗口核心安全成功率达到 0.8 后扩张。模式不健康时只提高该模式的 protected 回放比例；最终发布仍使用全指标、512 episode 和 0.9 通过率，不复用为课程门槛。

该阶段 PPO 初始学习率为 `5e-5`，并设置硬上限 `1e-4`；actor 高斯 std 限制在 `[0.05, 0.35]`。这限制了 KL 自适应调度抬升学习率以及熵奖励扩张探索方差的范围。
RSL-RL 在每次策略前向时对有效 std 执行 clamp，因此该范围是硬约束。优先观察 `Loss/learning_rate`、`Policy/mean_noise_std` 与 target profile 的速度误差趋势；只有高速能力停滞且旧技能 gate 仍通过时，才在下一 run 将 `std_max` 小幅提高到 `0.40`。

训练先按每个 cell 的 episode 成功率生成窗口，同时保留 35% 的旧技能复习配额；保护技能异常时自动提高到 50%。评测 profile 分为 baseline、target 和 diagnostic，只有 baseline + target 参与默认通过判定。

速度阶梯使用连续三档参数，而不是逐阶段独立试参：

| 速度范围 | 跟踪/欠速权重 | 高度/直立/俯仰 | 动作率/跌倒 | 主要目的 |
|---|---|---|---|---|
| `0.2–0.4 m/s` | `2.8–3.3 / 1.0–1.2` | `1.5–1.8 / 1.9–2.2 / 1.9–2.2` | `0.04 / 40` | 稳定起步、启停和零速保持 |
| `0.6–1.0 m/s` | `3.4–3.6 / 1.0` | `1.2–1.4 / 1.1–1.5 / 0.9–1.3` | `0.04–0.045 / 35–30` | 从姿态主导过渡到速度跟踪主导 |
| `1.2–2.0 m/s` | `3.7–4.2 / 1.1–1.4` | `1.0–1.15 / 0.8–1.0 / 0.6–0.8` | `0.05–0.06 / 30–25` | 高速跟踪、制动和轮地稳定 |

所有阶段都保留非零的航向、横向轨迹、轮滑、轮同步和姿态奖励；高速阶段只降低其相对权重，不再直接关闭。最低巡航速度比例统一为指令的 `85%`，速度门限和姿态门限随阶段编号单调放宽。

PPO 使用 `gamma=0.99`、`gae_lambda=0.95`、每阶段 4 个学习 epoch、`max_grad_norm=0.8`。学习率从低速的 `3e-4` 平滑降到最高速的 `1e-4`，熵系数从 `0.006` 降到 `0.004`，目标 KL 从 `0.015` 放宽到 `0.02`，避免原先在 1.2 m/s 附近突然降到 `5e-5`。

budget 是新增环境 transition 总数。默认每环境 rollout 为 24 步，迭代数为：

```text
ceil(budget / (num_envs × 24))
```

各阶段 episode 时长取该阶段最长的 profile。每个环境按编号轮流分配 profile，而不是随机丢 profile，从而保证 GPU batch 内覆盖全部运动模式。

## 8. 地形

- 平地阶段使用无限平面。
- 坡面地形包含起步平路、30° 上升段、坡顶和下降段。
- 原始坡面 JSON 的 5 m 起步距离和 8 m 坡长无法被其 25 s、约 3.58 m 的前进 profile 走到，因此坡面专项把上升段压缩到 3.5 m；30° 角度和 3 m 最低前进 gate 保持不变。
- 崎岖地形使用 0.01–0.04 m 连续高度场，训练难度按地形行递增，单点局部坡度限制在 30° 内。
- 障碍地形使用 2–5 cm 高、横跨两条轮迹的 box、cylinder 和 capsule；前向与反向任务都统计碾压通过次数。
- 综合任务依次组合平地、崎岖接近段、30° 坡、坡顶、下降、障碍和终点。地形长度扩展到 60 m，以容纳课程 profile 约 56.65 m 的理论前进距离。

地形与障碍使用独立碰撞层：机器人和地形可接触，机器人自身碰撞被屏蔽。

## 9. 奖励设计

奖励权重直接来自 `diablo_training/curriculum.json` 和 `flat_motion_speed_stages.json`，阶段 JSON 的 `reward_defaults` 与 `overrides` 优先于公共默认值。

正向形状项：

- 前进速度、低于目标速度、偏航速度和基座高度；
- 直立、俯仰、静止、位置保持漂移和角度保持漂移；
- 加速度、制动、原地转向；
- 轮速与机体速度匹配；

惩罚项：

- 目标路径法向速度、动作变化率、关节力矩和超过额定力矩；
- 同方向超速、关节软限位、轮滑、后滑；
- 左右轮中心偏差和直行轮速差；
- 垂直速度、横滚/俯仰角速度；
- 非回馈机械功率和跌倒。

事件奖励来自阶段 JSON：

- `obstacle_pass`；
- `slope_crest`；
- `finish`。

每回合同时累计速度/偏航 MAE、直立比例、俯仰和横滚、高度误差、轮滑、后滑、轮同步、停车速度、位置保持漂移、角度保持漂移、前进距离、启停次数、障碍数和终点状态。

速度阶梯还统计指令平台保持至少 `0.5 s` 后的有符号巡航速度、巡航速度 MAE 和巡航速度占指令峰值比例。`flat_motion_020` 要求巡航速度误差不超过 `0.02 m/s`，巡航速度至少达到指令的 `90%`，前进与后退使用同一门限，并分别在各 profile 中验证。

所有 `flat_motion_*` 和综合 `flat_motion` 阶段使用权重 `0.8` 的 Huber 超速惩罚。实际同方向速度超过指令 `5%` 后开始累计，用于抑制后向巡航持续偏快，同时保留速度跟踪修正空间。

所有训练阶段都使用目标路径法向速度惩罚横向漂移。对直线路径，该量等于与目标航向垂直的世界横向速度，既包含机体系横向速度，也包含“以错误航向向前行驶”产生的世界横向分量。Critic 额外接收横向位置误差、世界横向速度和路径法向速度；部署 Actor 的本体观测维度保持不变。

三个航向相关指标含义不同：

- `yaw_mae_rad_s` 是整回合平均偏航角速度跟踪误差。速度阶梯阶段 gate 为 `0.02 rad/s`（约 `1.15°/s`），奖励容差为 `0.01 rad/s`；复杂地形阶段保留各自门限。
- `heading_drift_after_0.5s_deg` 是每次静止保持完成 `0.5 s` 建立后，该保持窗口内相对建立时刻的最大角度漂移。训练使用其 Huber 损失；有硬门限时容差取门限的 `75%`，没有门限时回退到 `heading_tolerance_rad`。
- `max_hold_drift_m` 是位置漂移，单位为米，不能用于替代角度漂移。

速度阶梯的俯仰门限随速度缓升，最大俯仰统一为 `10°`：

| 速度阶段 | 平均/稳态 pitch | `min_upright` |
|---|---:|---:|
| `0.1–0.4 m/s` | `3°` | `0.93` |
| `0.6–1.0 m/s` | `3.25°` | `0.91` |
| `1.2–1.6 m/s` | `3.5°` | `0.89` |
| `1.8 m/s` | `4°` | `0.87` |
| `2.0 m/s` | `4.25°` | `0.86` |

`upright = exp(-(roll² + pitch²) / 0.04)` 同时约束横滚和俯仰。表中 upright 门限对应约 `3.1°–3.7°` 的纯姿态等效 RMS，为高速瞬态和横滚保留余量。

速度阶梯阶段的轮滑拆成两项：

- `mean_steady_wheel_slip_mps` 只统计移动、直行、指令平台保持至少 `0.5 s` 后的轮面速度与机体速度差，排除起步、加减速和指令切换。
- `transient_wheel_slip_mps` 是整体启动宽限后的最大轮滑，用于发现冲击和短暂失配。

轮滑百分比继续作为诊断日志保留，但速度阶梯不再同时用绝对值和百分比形成互相冲突的硬门限。`flat_balance` 继续统计 `1.0 s` 后的稳态指标；其他平地阶段使用 `0.5 s` 启动宽限。

训练日志把自定义总奖励拆为 `Reward/height`、`Reward/posture`、`Reward/stationary`、`Reward/tracking`、`Reward/wheel_quality`、`Reward/effort` 和 `Reward/fall`，各组之和为 `Reward/total`。

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

从 080 开始运行独立转向主干课程：

```bash
diablo course-train \
  --course turning \
  --warm-start logs/rsl_rl/diablo_xingtian_course/<080-run>/model_0.pt \
  --critic-calibration-steps 98304 \
  --n-envs 4096 \
  --gpu-ids 0 \
  --log-root logs/rsl_rl
```

`course-train --course turning` 只运行 7 个主干阶段，并对最后阶段同样执行 gate。

运行自适应速度扩展阶段：

```bash
diablo train \
  --stage flat_adaptive_speed \
  --warm-start logs/rsl_rl/diablo_xingtian_course/<turn-run>/model_<iter>.pt \
  --critic-calibration-steps 98304 \
  --n-envs 4096 \
  --gpu-ids 0 \
  --log-root logs/rsl_rl
```

如果现有策略在同一个 checkpoint 中已经保留平衡、0.8 m/s 直行和原地转向，直接从这个 checkpoint 热启动；不需要同时合并多个 checkpoint。

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

续训会比较当前阶段实际消费的训练指纹，包括动作/观测接口、该阶段课程、奖励、终止、随机化和 PPO 配置，以及模型、控制与任务代码哈希。修改其他阶段或仅修改评估 gate 不会阻止当前阶段续训；修改当前阶段训练语义仍会要求新建 run。

旧版 run 的 `contract.json` 没有阶段训练指纹，仍使用保守的全源码哈希回退；这类 run 在升级后需要重新生成一次。

跨阶段训练只复用旧策略，不继承旧 critic、优化器或训练迭代状态：

```bash
diablo train \
  --stage flat_motion_020 \
  --warm-start logs/rsl_rl/diablo_xingtian_course/<run>/model_<iter>.pt \
  --transfer actor \
  --initial-std 0.3
```

该命令接受与当前 `temporal_residual_v1` 架构一致的 RSL-RL `.pt` checkpoint。旧 MLP checkpoint、`diablo-rsl-actor-v1` actor bundle 和外部 SB3 `.zip` 不再用于新策略热启动。`--warm-start` 与 `--resume` 互斥，并始终创建新 run、新 optimizer 和迭代 0。
`flat_adaptive_speed` 不再限制必须来自 720 或任何特定源阶段；任意通过 actor 接口校验的 RSL-RL checkpoint 都可以启动新 run。

来自本项目 run 的权重会同时校验接口和策略架构契约；跨课程阶段复用 actor 时必须形如本项目的统一时序残差网络。

可选的 critic 迁移：

```bash
diablo train \
  --stage flat_motion_020 \
  --warm-start logs/rsl_rl/diablo_xingtian_course/<run>/model_<iter>.pt \
  --transfer actor-critic \
  --initial-std 0.3 \
  --critic-calibration-steps 5000
```

- `--transfer actor` 只复制 actor，critic 和 optimizer 全新初始化。
- `--transfer actor-critic` 要求原生 RSL-RL checkpoint，额外复制 critic，但仍拒绝继承源 optimizer 和迭代。
- `--initial-std` 覆盖复制后 actor 的动作标准差，用于恢复跨阶段探索。
- `--critic-calibration-steps` 冻结 actor，用新阶段环境只校准 critic；实际按完整 PPO rollout 向上取整，请求值、实际迭代数和 transitions 写入 checkpoint 的 `infos.actor_init`。

`course-train` 同样接收 `--transfer`、`--initial-std` 和 `--critic-calibration-steps`，阶段之间默认使用 actor-only warm-start。

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
  --stage composite \
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
  --seeds 2001,2002,2003 \
  --eval-mode fast
```

`fast` 在单个 MJLab 向量环境中并行运行不同 profile，并至少给每个 profile 分配 32 个 episode；`exact` 保留同 seed/profile 的严格配对语义，用于最终复核。`--profile-set` 可选 `required`、`all`、`baseline`、`target` 和 `diagnostic`。自适应 run 使用 `select-best --stage flat_adaptive_speed`。

导出任意 checkpoint：

```bash
diablo export \
  --stage composite \
  --checkpoint <run>/policy-best.pt \
  --output runs/final-export
```

ONNX 的 Actor 输入为 `float32[N,125]`，输出为 `float32[N,6]`。导出成功只证明网络与接口一致；实机部署仍需零位、电机驱动、通信延迟、负载和安全性标定。

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

完整课程包含约 226M transition。CI 中不执行完整训练，也不把短训练结果作为运动能力证据。

### 交付检查记录

2026-10-28 在 Windows 源码工作区完成：

- `python -m compileall` 通过；
- 14 项课程、gate、评分和 CPU 模型测试全部通过；
- 使用 MuJoCo 3.11.0 和锁定公开资源验证主分支 CAD 基线、六执行器、两个闭环、正向轮动作、FK、虚功和重置；
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
