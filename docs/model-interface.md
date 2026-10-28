# 模型与策略契约（v3）

## 模型来源与范围

模型资源固定在 `upstream.lock.json`：

- `DDTRobot/Col_DIABLO_Issac_RL` 提供 CAD URDF、惯量和网格；
- `DDTRobot/diablo-sim-env` 提供 Webots 串联髋膝、回程支链和闭环关系；
- DIABLO 论文提供机构与 M1502D 电机参考。

MJLab 和 CPU 检查复用同一个 `make_robot_spec()`。机器人包含六个主动电机、四个被动关节、两个 `connect` 闭环和浮动基座。

CAD 原始质量分布在加载时统一缩放到公开产品页的 22.9 kg。该修正只调整总质量，不声称已经按实物称重重新分配各连杆质心。

| 参数 | 数值 |
|---|---:|
| 主动关节 | 6 |
| 被动关节 | 4 |
| 闭环 | 2 |
| 官方总质量 | 22.9 kg |
| 轮半径 | 0.1 m |
| 堵转力矩 | 17 Nm |
| 额定力矩 | 9.6 Nm |
| 空载速度 | 23.038 rad/s |
| 物理步长 | 0.002 s |
| 控制步长 | 0.01 s |

训练基座目标高度为 `0.270943 m`，整机课程高度为 `0.37958 m`。官网另一个 491 mm 整机尺寸属于不同产品版本，不作为本 CAD 模型的覆盖值。

## 动作

动作固定六维、float32、范围 `[-1,1]`：

| 索引 | 关节 | 控制语义 |
|---:|---|---|
| 0 | `left_hip_motor` | 默认位置 + `0.5a` |
| 1 | `left_knee_motor` | 默认位置 + `0.5a` |
| 2 | `left_wheel_motor` | 轮速目标，正动作前进 |
| 3 | `right_hip_motor` | 默认位置 + `0.5a` |
| 4 | `right_knee_motor` | 默认位置 + `0.5a` |
| 5 | `right_wheel_motor` | 轮速目标，正动作前进 |

默认原始关节位置为：

```text
[0.5, -1.0, 0.0, 0.5, -1.0, 0.0] rad
```

髋、膝使用 MJLab 原生位置执行器，刚度/阻尼分别为 15/0.3 和 18/0.2，力矩上限 17 Nm。

轮执行器使用 GPU 张量 PI：

```text
target_slew = 20 rad/s²
error = target_wheel_speed - measured_wheel_speed
integral_error 限制为 2 Nm / ki
torque = 0.05 * target + kp * error + ki * integral
```

默认 `kp=0.3`、`ki=4.0`；速度阶段可从课程 JSON 覆盖目标轮速上限。输出再通过和现有 CPU 控制一致的线性转速-力矩包络。

## Actor 观测

Actor 输入固定为 `[B,160]`。

### 本体历史：125

每帧 25 维：

| 切片 | 内容 | 缩放 |
|---|---|---|
| `0:3` | 机体系角速度 | 0.25 |
| `3:6` | 机体系投影重力 | 1 |
| `6:9` | `[vx, yaw_rate, base_height]` 指令 | `[10, 0.25, 5]` |
| `9:13` | 左髋、左膝、右髋、右膝相对默认位置 | 1 |
| `13:19` | 六电机速度；轮速度反号以正动作前进 | 0.05 |
| `19:25` | 上一时刻归一化动作 | 1 |

历史长度为 5，按最旧到最新拼接，共 125 维。

### 地形扫描：35

基座前方覆盖约 `1.5 m × 1.0 m`，使用垂直向下的 `7×5` 射线。返回值为相对基座的高度，减去 `0.270943 m` 基准后除以 3 m 范围。平坦地面的中心值接近零。

### Critic

Critic 复用 Actor 输入，并增加：

- 真实基体线速度 3；
- 真实基座高度 1；
- 地形法向 3；
- 轮地接触命中均值 1。

特权信息只用于训练价值函数，不追加到 Actor，不导出到 ONNX。

## 指令与事件时序

课程 profile 由 `[时间, 前进速度, 偏航速度]` 点线性插值。每控制步：

1. 应用当前动作并推进 5 个物理子步；
2. 使用当前 profile 指令计算奖励、gate 指标和事件；
3. 推进 episode 计数器；
4. 更新下一时刻指令和观测。

每回合从时间零开始，每个 profile 固定分配而不是中途随机重采样。方向为负的前进指令表示后退；`start="end"` 的 profile 在对应地形末端初始化。

## 地形

- 平地任务使用 MuJoCo plane。
- 坡面、崎岖、障碍和综合任务使用 MJLab 高度场。
- 崎岖表面连续变化，幅度从 0.01 m 逐级增加到课程上限 0.04 m，局部坡度不得超过 35°。
- 障碍为 0.02–0.05 m 高、横跨双轮迹的 box、cylinder 或 capsule。
- 地形碰撞层 `contype=2/conaffinity=1`，机器人碰撞层 `contype=1/conaffinity=2`，因此机器人接地但不与自身碰撞。

## 保存与兼容性

每个 run 的 `contract.json` 包含：

- MJLab、MuJoCo、后端版本；
- 模型目标高度和动作顺序；
- `[160]` 观测与 `[6]` 动作；
- 18 阶段顺序和任务 ID；
- 初筛、复筛、盲测、边界 seed；
- 完整内容 SHA-256。

checkpoint 本身由 RSL-RL 保存。policy-best 筛选会先验证 run contract，再逐一加载 `model_*.pt`。不同观测形状、动作顺序、模型高度或阶段契约不能强行通用。

最终筛选只在 `composite` run 中进行：

1. `2001–2003` 初筛，所有 seed/profile 必须全部通过；
2. `2004–2008` 复筛，所有 seed/profile 必须全部通过；
3. 按综合 gate 分数排序；
4. 保存 `policy-best.pt`、`policy-best.onnx` 和 `selection.json`；
5. `3101–3110` 和 `4101–4103` 仅做选择后报告。

## 导出

ONNX 输入名为模型自身生成值，固定形状为 `[N,160]`，输出为 `[N,6]`，opset 18。文件元数据包含阶段、checkpoint 哈希、interface 哈希、控制周期和以下边界：

```text
simulation_only=true
hardware_authorized=false
```

导出前会加载 checkpoint 并运行 MJLab Runner；失败不会生成一个仅凭维度猜测可用的模型。
