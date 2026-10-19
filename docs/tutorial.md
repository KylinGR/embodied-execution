# DIABLO MuJoCo 训练：从安装到第一份策略

目标是让你实际走完：**安装 → 检查物理模型 → 训练 → 保存 → 重载回放 → 导出**。第一轮先确认工具链能工作，随后才增加训练预算和判断运动效果。

第一版只做平地基础运动，不需要购买机器人或接通电机。使用的是上游训练用等效串联模型，尚未标定真实并联机构；详见[模型契约](model-interface.md)。

## 1. 准备 Python

本教程面向 Linux。已验证的开发环境是 Python 3.11；也允许 Python 3.12，但不能把未运行过的组合当作已验证。

```bash
python3 --version
git --version
```

如果系统 Python 不是 3.11/3.12，可使用你已有的 Conda：

```bash
conda create -n embodied-execution python=3.11 -y
conda activate embodied-execution
```

没有 Conda 时，先安装 Python 3.11/3.12 及其 venv 支持。不同 Ubuntu 发行版的软件包不同，不要直接覆盖系统 Python。

## 2. 安装项目与获取原始模型

```bash
git clone https://github.com/KylinGR/embodied-execution.git
cd embodied-execution
bash scripts/setup.sh
source .venv/bin/activate
```

脚本在当前仓库建立 `.venv`；没有 PyTorch 时先安装 CPU 版，然后安装项目、获取固定提交的原始模型并运行预检。首次执行需要访问 PyPI、PyTorch 下载站和 GitHub。

如果你已激活一个合适的 Python 环境，也可以逐步操作：

```bash
python -m pip install 'torch>=2.8,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
diablo fetch
diablo doctor
```

`fetch` 只克隆资源，不安装或导入上游 Isaac Gym 代码。固定提交见根目录 `upstream.lock.json`。若已有资源的版本或内容被修改，它会报错，不会自动重置你的文件。

## 3. 检查“物理模型能不能运行”

```bash
diablo doctor
diablo smoke --steps 200 --output runs/my-smoke
```

`doctor` 应显示六个执行器、125维观测、6维动作，以及 `gymnasium_check: passed`。它检查的是环境接口，不是运动能力。

`smoke` 真正推进 MuJoCo，施加小幅随机动作，保存 `trace.jsonl` 与 `result.json`。200 个控制步累计为 2 秒仿真，**可能包含多个回合和跌倒后的训练复位**，不代表连续站立两秒。每个结束回合都会记录，不隐藏失败。

这一步若出现非有限状态、数值警告或资源缺失，应先解决；先不要投入长训练。

## 4. 第一次训练：小预算验证

```bash
diablo train --steps 2048 --n-envs 4 --output runs/first-train
```

程序复用 Stable-Baselines3 PPO，默认 CPU、单个 PyTorch 计算线程。四个环境使用 MuJoCo CPU 物理；这不是 GPU 并行物理训练。

每个环境的 rollout 为128步，一次更新采集 `128 × n_envs` 个 transition。因此请求的步数会向完整 rollout 取整。`result.json` 同时记录请求步数、实际步数、耗时及策略参数是否改变。

训练目录中的主要文件：

| 文件 | 用途 |
|---|---|
| `resolved.json` | 环境、PPO、模型来源、版本和有效参数 |
| `policy.zip` | 可加载的完整 PPO 检查点 |
| `result.json` | 实际步数、时间、参数更新证据 |
| `episodes.csv.monitor.csv` | 各回合长度与奖励；具体名称由 SB3 Monitor 追加后缀 |
| `tensorboard/` | 学习曲线 |
| `checkpoints/` | 每100,000个累计新增 transition 附近保存的定期检查点 |
| `failure.json`、`failure-policy.zip` | 发生异常或中断时保留的失败信息和已有参数 |

短训练完成只说明参数更新和保存可用。奖励升高也不等于行驶、制动、保持已经合格。

## 5. 加载策略并观察效果

```bash
diablo evaluate --model runs/first-train/policy.zip --output runs/first-evaluate
```

它分别运行四个基础任务的首个回合，不对跌倒回合自动重试。四个任务分别是：

1. 原地平衡。
2. 低速起步、巡航、减速，10秒以后保持。
3. 0 → 低速 → 较高速 → 低速 → 0，16秒以后保持。
4. 同一回合完成两次起步与停车，16秒以后保持。

每回合最长20秒；策略跌倒或身体触地就提前结束。退出码 `2` 表示有物理失败或查看窗口被关闭，**不是安装失败**。结果和原始轨迹仍会保存。

结果包括实际回合时长、速度误差、末速度、最后半秒最大前向速度、保持漂移。未运行到保持阶段时，对应指标为 `null`，不能把它记成零漂移。

### 录视频

有可用 EGL 的无显示器 Linux 主机：

```bash
MUJOCO_GL=egl diablo play --model runs/first-train/policy.zip \
  --video --output runs/first-video
```

视频保存为 `rollout.mp4`，25 fps；原始轨迹为 `trajectory.npz`，包含导出验证可直接使用的观测和动作。

有桌面显示环境时可以打开交互窗口：

```bash
diablo play --model runs/first-train/policy.zip --viewer --output runs/first-viewer
```

`--viewer` 按仿真时间放慢显示。关闭窗口会结束评估并记录取消，不会继续自动运行。

## 6. 增加预算与续训

先检查回放、奖励组成和动作；确认没有模型或接口错误后再增加预算：

```bash
diablo train --steps 200000 --n-envs 4 --output runs/foundation-200k
```

或从上一份策略继续更新：

```bash
diablo train --resume runs/first-train/policy.zip \
  --steps 200000 --n-envs 4 --output runs/foundation-resume
```

续训继承检查点环境配置，输出写入新的目录。保存了模型和优化器，但物理回合重新开始，不承诺随机数和物理帧级精确恢复。

初期可把速度上限降为0.2 m/s：

```bash
diablo train --max-speed 0.2 --steps 200000 --n-envs 4 --output runs/foundation-low-speed
```

`--noise` 只加入小幅陀螺仪和编码器速度噪声，不代表完整动力学随机化或 sim2real 验证。第一版速度上限限制在0.5 m/s；更复杂任务应先开发对应环境和评测。

查看曲线：

```bash
tensorboard --logdir runs/foundation-200k/tensorboard
```

重点看回合是否变长、是否到达运动与保持阶段、实际速度误差是否下降。连续两轮同类改动没有改善时，回看原始状态、接触与动作映射，不继续盲扫相近奖励权重。

## 7. 导出并核对动作

```bash
diablo export --model runs/first-train/policy.zip --output runs/first-export
```

也可以用实际回放观测：

```bash
diablo export --model runs/first-train/policy.zip \
  --observations runs/first-evaluate/trajectory.npz --output runs/export-from-replay
```

输出 `policy.onnx`、`policy.pt`、`contract.json`。导出器会真正加载 ONNX Runtime 和 TorchScript，对同一批最多256条观测比较动作；默认自行采集128条物理观测。容差为 `atol=rtol=1e-5`，失败会报错。

**导出成功只证明动作数值一致。** 模型输入是本项目定义的 `[N,125]` 历史观测，并非厂家 D1h、其他 DIABLO 项目或实机 SDK 的通用接口。不要连接电机测试。

## 8. 常见问题

| 现象 | 处理 |
|---|---|
| `Use Python 3.11 or 3.12` | 切换 Python 环境后重新执行安装；不要覆盖系统 Python |
| `Model source is missing` | 执行 `diablo fetch` |
| 上游版本或文件被修改 | 保留你的改动，另用干净克隆；不自动 reset |
| 输出目录已存在 | 改用新的目录；续训也使用新的输出目录 |
| EGL／GLFW 报错 | 先不加 `--video/--viewer` 完成训练检查，再配置图形驱动；纯软件渲染可安装 `libosmesa6` 并使用 `MUJOCO_GL=osmesa` |
| 刚训几千步就跌倒 | 这是策略效果问题；看回合时长、接触和奖励，不把退出码2当依赖错误 |
| MuJoCo 非有限状态／数值警告 | 保留实验目录，检查模型和控制参数；程序不接受自动数值复位作为成功 |
| 显存占用低 | 默认 CPU 物理与 CPU 策略；GPU利用率不是本项目的验收指标 |
| `resolved.json must accompany...` | 使用本项目训练目录下的检查点，并保留其配置文件 |

## 9. 六人团队如何协作

| 负责人 | 首轮任务 |
|---|---|
| 集成 | 固定模型接口和完整验收命令，处理主阻塞 |
| 模型 | 对照实物资料、并联腿运动学和执行器，逐步替换当前近似 |
| 环境 | 维护 `env.py` 的课程、奖励和失败定义 |
| 训练 | 使用同一基线训练，管理预算和检查点 |
| 部署接口 | 核对观测时序与导出契约，先做离线推理 |
| 评估 | 保存首回合失败，独立比较速度、制动和保持表现 |

所有人共享一套模型和接口。先完成基础运动，再讨论复杂地形和跳跃。
