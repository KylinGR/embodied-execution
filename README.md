# 具身执行 · Embodied Execution

面向团队的 **DIABLO 双轮足机器人 MuJoCo 运动训练入门项目**。

本仓库开发了 MuJoCo 模型适配、Gymnasium 环境、基础运动课程、PPO 训练、回放和策略导出。PPO 复用 Stable-Baselines3，无需 Isaac Gym、ROS 或机器人硬件。默认使用 CPU；不连接实机电机。

![MuJoCo中的DIABLO模型初始姿态](docs/images/diablo-mujoco.png)

图为物理模型初始姿态，不是已训练策略的稳定运动演示。

**模型边界：**第一版使用厂家推荐强化学习示例中的六自由度等效串联 URDF，保留其几何、质量和惯量。它不是 DIABLO 实机并联腿闭环及执行器的标定模型，训练结果不代表实机能力。本仓库不附带成熟的预训练运动策略。

## 从这里开始

需要 Linux、Git、Python **3.11 或 3.12**。基础训练不要求 NVIDIA GPU。完整操作和报错处理见[新手教程](docs/tutorial.md)。

```bash
git clone https://github.com/KylinGR/embodied-execution.git
cd embodied-execution
bash scripts/setup.sh
source .venv/bin/activate

# 实际创建 MuJoCo 环境并检查接口
diablo doctor
diablo smoke --steps 200

# 先验证训练链；2,048 步不够证明已经学会运动
diablo train --steps 2048 --n-envs 4 --output runs/first-train

# 不录视频时不需要图形环境；发生跌倒会保留结果并返回退出码 2
diablo evaluate --model runs/first-train/policy.zip --output runs/first-evaluate

# 导出并实际比较 ONNX、TorchScript 和原策略的动作
diablo export --model runs/first-train/policy.zip --output runs/first-export
```

每次实验使用新的输出目录，已存在时拒绝覆盖。默认课程为：平衡、单次起停、行驶变速、重复起停；最高指令速度 0.5 m/s，末段保持。当前不包含复杂地形、转向课程、跳跃或实机部署。

## 教程与资源

- [安装、首轮训练、回放、续训、看曲线](docs/tutorial.md)
- [模型、观测、动作、时钟和导出契约](docs/model-interface.md)
- [官方资源及上游来源](docs/resources.md)
- [许可证和资源再分发边界](THIRD_PARTY.md)

## 本地验证

2026-10-19，Linux / Python 3.11.16 / MuJoCo 3.8.1 / Gymnasium 1.4.0 / SB3 2.9.0 / PyTorch 2.11.0，CPU策略与CPU物理：

| 检查 | 实际结果 |
|---|---|
| Gymnasium及物理/接口检查 | 环境检查与5项测试通过 |
| 小幅随机动作预检 | 累计200控制步／2秒仿真；发生一次正常回合终止并保留 |
| PPO训练 | 实际2,048个transition，墙钟2.02秒，参数SHA变化，检查点保存成功 |
| 保存重载及EGL视频 | 四个任务均在0.66秒跌倒，尚未进入起步阶段；失败原样保留 |
| 策略导出 | 128条物理观测，ONNX最大动作误差8.94e-8，TorchScript误差0 |
| 续训 | 新增512个transition，参数继续更新；TensorBoard写入新实验目录 |

这些结果证明训练、保存、回放与导出链路可运行，**尚未证明学会稳定平衡、行驶或停车**；模型未实机标定。上述耗时来自本机短任务，不是性能保证或长时可靠性测试。完整日志、检查点和失败视频保留在本地`runs/validation-*`，未作为预训练模型发布。

## 项目结构

```text
diablo_training/model.py       原始 URDF → MuJoCo、传感器、执行器
diablo_training/env.py         物理环境、25维观测、5帧历史、基础课程
diablo_training/cli.py         预检、训练、回放、导出
upstream.lock.json            固定原始资源提交与用途
scripts/setup.sh              安装与模型资源获取
tests/                       物理及接口检查
docs/                        中文教程
runs/                        本地实验与原始失败记录，不提交
third_party/                 安装时下载的原始资源，不提交
```

仓库公开以方便团队获取和审查；不要将账号凭据、设备地址、内部数据或实机授权写进配置。
