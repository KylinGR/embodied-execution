# XINGTIAN A/B/C/D 基础运动课程

本课程只覆盖基础运动。A 使用平地任务，B/C/D 使用 `XingtianStageEnv`。四个阶段保持
125 维观测和 6 维动作不变，因此可以只加载策略网络权重进行 warm-start。

## 1. 阶段和配置

| 阶段 | 配置 | 内容 |
|---|---|---|
| A | `examples/xingtian_reference.json` | 平地平衡、前进、后退、转向、启停 |
| B1 | `examples/xingtian_stage_b_l1.json` | 崎岖 Level 1 |
| B2 | `examples/xingtian_stage_b_l2.json` | 崎岖 Level 2 |
| B3 | `examples/xingtian_stage_b_l3.json` | 崎岖 Level 3 |
| B4 | `examples/xingtian_stage_b_l4.json` | 崎岖 Level 4 |
| C1 | `examples/xingtian_stage_c_l1.json` | 20 mm 障碍 |
| C2 | `examples/xingtian_stage_c_l2.json` | 30 mm 障碍 |
| C3 | `examples/xingtian_stage_c_l3.json` | 40 mm 障碍 |
| C4 | `examples/xingtian_stage_c_l4.json` | 50 mm 障碍 |
| D | `examples/xingtian_stage_d_mixed.json` | 崎岖地形和障碍混合 |

`--resume` 用于同一环境继续训练；`--warm-start` 用于任务/奖励/地形改变时只加载网络权重。

## 2. A 阶段

```bash
diablo train \
  --config examples/xingtian_reference.json \
  --device cuda:0 \
  --vec-env subproc \
  --n-envs 16 \
  --steps 2000000 \
  --output runs/curriculum/A
```

选择 A 的最佳 checkpoint：

```bash
diablo select-best \
  --run runs/curriculum/A \
  --stages stand forward backward turn start_stop \
  --seed 2001
```

最佳权重位于：

```text
runs/curriculum/A/selection/policy-best.zip
```

## 3. B 阶段

B1 从 A 最佳权重 warm-start：

```bash
diablo train \
  --config examples/xingtian_stage_b_l1.json \
  --warm-start runs/curriculum/A/selection/policy-best.zip \
  --device cuda:0 \
  --vec-env subproc \
  --n-envs 16 \
  --steps 1500000 \
  --output runs/curriculum/B1

diablo select-best \
  --run runs/curriculum/B1 \
  --stages forward backward turn start_stop \
  --seed 2002
```

B2～B4 使用前一阶段最佳权重：

```bash
diablo train \
  --config examples/xingtian_stage_b_l2.json \
  --warm-start runs/curriculum/B1/selection/policy-best.zip \
  --device cuda:0 --vec-env subproc --n-envs 16 \
  --steps 1500000 --output runs/curriculum/B2

diablo train \
  --config examples/xingtian_stage_b_l3.json \
  --warm-start runs/curriculum/B2/selection/policy-best.zip \
  --device cuda:0 --vec-env subproc --n-envs 16 \
  --steps 1500000 --output runs/curriculum/B3

diablo train \
  --config examples/xingtian_stage_b_l4.json \
  --warm-start runs/curriculum/B3/selection/policy-best.zip \
  --device cuda:0 --vec-env subproc --n-envs 16 \
  --steps 2000000 --output runs/curriculum/B4
```

每个 B 级都在训练后运行：

```bash
diablo select-best \
  --run runs/curriculum/B4 \
  --stages forward backward turn start_stop \
  --seed 2002
```

将路径中的 `B4` 替换为当前级别，逐级验收。

## 4. C 阶段

C1 从 B4 最佳权重开始：

```bash
diablo train \
  --config examples/xingtian_stage_c_l1.json \
  --warm-start runs/curriculum/B4/selection/policy-best.zip \
  --device cuda:0 --vec-env subproc --n-envs 16 \
  --steps 1000000 --output runs/curriculum/C1

diablo select-best \
  --run runs/curriculum/C1 \
  --stages forward turn start_stop \
  --seed 2003
```

C2～C4 按相同方式逐级 warm-start。C 场景包含左右走廊边界和横跨路线的显式障碍，
不应通过大幅转向绕过障碍。

## 5. D 阶段

D 从 C4 最佳权重开始：

```bash
diablo train \
  --config examples/xingtian_stage_d_mixed.json \
  --warm-start runs/curriculum/C4/selection/policy-best.zip \
  --device cuda:0 --vec-env subproc --n-envs 16 \
  --steps 3000000 --output runs/curriculum/D

diablo select-best \
  --run runs/curriculum/D \
  --stages stand forward backward turn start_stop \
  --seed 2004 \
  --min-success 1.0
```

## 6. 选择规则

`select-best` 会：

1. 对 `checkpoints/*.zip` 和最终 `policy.zip` 分别评测；
2. 使用固定 seed 和脚本化指令；
3. B/C/D 评测时使用与训练不同的地形 seed；
4. 记录成功率、速度误差、偏航误差和最小 upright；
5. 优先选择达到 `--min-success` 的候选；
6. 将最佳 checkpoint 复制为 `selection/policy-best.zip`；
7. 同时复制 `resolved.json` 并写入 `selection.json`。

分数越低越好。最终盲测不能用于选择 checkpoint。

## 7. 阶段门槛

- A 未通过平地平衡、启停和多次启停，不进入 B；
- B 未通过当前 Level，不升级到下一 Level；
- C 未通过当前障碍高度，不提高障碍高度；
- D 必须同时回归 A/B/C；
- 任一阶段出现只靠拖底、碰撞墙或侧向滑移通过，不得放行。
