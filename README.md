# 具身执行 · Embodied Execution

面向团队的 **DIABLO 双轮足机器人 MuJoCo 运动训练环境**。提供公开机构模型适配、Gymnasium环境、训练/续训、评估、可视化和ONNX/TorchScript导出。默认CPU，无需Isaac Gym、ROS或机器人硬件。你可以自行定义任务、奖励和训练策略。

![MuJoCo中的DIABLO模型初始姿态](docs/images/diablo-mujoco.png)

图为物理模型初始姿态，不是已训练策略演示。

v0.2已按公开CAD、官方Webots和论文校核串联髋膝电机、平行四边形闭环、电机轴向及轮子前进方向；修正机体速度坐标。**这是公开参考仿真，不是已标定实机**：质量/轮径与论文存在版本差异，未获得你的硬件零位及驱动参数。没有附带成熟运动策略。

## 开始训练

新增实验：[平稳落地训练、独立验收与已训练模型](docs/stable-landing.md)。当前反馈＋PPO 残差策略通过约 2–5 cm 仿真自由落地采样验收；不代表主动跳跃或实机能力，默认平地任务保持原状。

需要Linux、Git和Python **3.11或3.12**。不要求NVIDIA GPU；完整步骤见[中文教程](docs/tutorial.md)。

```bash
git clone https://github.com/KylinGR/embodied-execution.git
cd embodied-execution
bash scripts/setup.sh
source .venv/bin/activate

# 机构、驱动方向、传感器及Gymnasium接口校核
diablo doctor

# 首轮短训练，只确认工具链
diablo train --steps 2048 --n-envs 4 --output runs/first-train

# 每个任务运行首个回合；跌倒保留证据，退出码2
diablo evaluate --model runs/first-train/policy.zip --output runs/first-evaluate

# 用实际回放观测验证导出的动作
diablo export --model runs/first-train/policy.zip \
  --observations runs/first-evaluate/trajectory.npz --output runs/first-export
```

每次输出目录必须是新的。默认平地任务为平衡、单次起停、变速、重复起停。当前不提供越野、跳跃、爬行课程或硬件部署；这些需要另写任务和对应评测。

## 自己定训练策略

```bash
# 直接力矩动作及PPO参数由JSON配置，不依赖默认PD动作
diablo train --config examples/torque.json --steps 2048 --output runs/torque-train

# 改任务和奖励：继承环境，物理、传感器、时序复用
export PYTHONPATH="$PWD/examples${PYTHONPATH:+:$PYTHONPATH}"
diablo train --config examples/custom.json --steps 2048 --output runs/custom-train
```

[自定义教程](docs/tutorial.md#自定义任务奖励和算法)说明如何替换奖励、指令、网络及算法。基础Gymnasium环境可直接交给其他RL算法。模型/控制/观测指纹会拒绝不兼容检查点，包括旧版v0.1。

## 教程与资源

- [安装、训练、回放、续训、看曲线、自定义与排错](docs/tutorial.md)
- [模型来源、机构、电机映射、输入输出和时序](docs/model-interface.md)
- [官方资源、开源策略与SDK边界](docs/resources.md)
- [许可证与资源再分发边界](THIRD_PARTY.md)

## 实际验证

可重复的完整检查命令：

```bash
bash scripts/check_training.sh
```

它创建新的`runs/check-*`，集中完成测试、机构检查、随机动作、训练、续训、四任务首回合评估、实际轨迹导出。评估中的跌倒如实记录，不作为环境安装故障或运动成功。GitHub Actions还检查全新依赖安装和离开源码目录后的wheel安装。

2026-10-19，Linux / Python3.11.16 / MuJoCo3.8.1 / Gymnasium1.4.0 / SB3 2.9.0 / PyTorch2.11.0，CPU：

| 检查 | 实际结果 |
|---|---|
| 机构独立FK、闭环、虚功 | 12组姿态；最大误差分别1.33e-7 m、1.22e-16 m、2.55e-9 Nm |
| 物理/坐标/时序/接口/故障检查 | 10项测试和Gymnasium检查通过 |
| PPO→续训 | 2,048+512 transition，墙钟约2+1秒；两次参数均更新 |
| 重载首回合评估 | 四任务均0.75秒跌倒，尚未进入起步；总计3秒仿真 |
| ONNX/TorchScript | 256条实际仿真观测；最大动作误差1.04e-7 / 0 |

结果证明环境与训练流程可运行，**不证明已学会稳定运动，也不是长时验证**。原始日志和失败轨迹保留于本地`runs/check-20261019T034506-YZ3GIT`；CI短验收证据由Actions保留14天，不作为预训练权重发布。Python3.12允许安装，本地实测为3.11。

## 项目结构

```text
diablo_training/model.py          CAD → MuJoCo、闭环、IMU、执行器
diablo_training/control.py        电机坐标、PD/力矩动作、转速包络
diablo_training/env.py            Gymnasium、传感观测、历史与任务钩子
diablo_training/validation.py     独立公开机构和驱动校核
diablo_training/contract.py       检查点兼容校验
diablo_training/cli.py            训练、续训、回放、导出
diablo_training/upstream.lock.json 固定原始资源版本
examples/                        配置和自定义任务示例
scripts/                         安装与完整流程检查
tests/                           物理和接口回归检查
docs/                            中文教程
runs/                            实验和失败证据，不提交
third_party/                     原始模型仓库，不提交
```

仓库公开，方便团队获取和审查；内部数据与设备凭据不随仓库分发。
