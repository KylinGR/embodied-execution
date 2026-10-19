# DIABLO MuJoCo 训练：从安装到第一份策略

目标是让你实际走完：**安装 → 检查物理模型 → 训练 → 保存 → 重载回放 → 导出**。第一轮先确认工具链能工作，随后才增加训练预算和判断运动效果。

默认只做平地基础任务，不需要购买机器人或接通电机。v0.2已按官方公开资料恢复串联电机与平行四边形闭环；几何映射通过独立检查，但未实机标定。详见[模型契约](model-interface.md)。

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

`fetch` 只克隆资源，不安装或导入上游 Isaac Gym 代码。固定提交见 `diablo_training/upstream.lock.json`。两份资源分别提供CAD和官方Webots机构参考。若已有资源的版本或内容被修改，它会报错，不会自动重置你的文件。

## 3. 检查“物理模型能不能运行”

```bash
diablo doctor
diablo smoke --steps 200 --output runs/my-smoke
```

`doctor` 应显示6个执行器、2个闭环、125维观测、6维动作，以及 `gymnasium_check: passed`、`mechanics.checks: passed`。它会独立对比官方机构变换、校验驱动与虚功；这不是运动能力测试。

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
| `resolved.json` | 环境、任务类、PPO、模型来源、版本和兼容指纹 |
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

它分别运行五个基础任务的首个回合，不对跌倒回合自动重试。五个任务分别是：

1. 原地平衡：全程保持零前进和零偏航指令。
2. 单次启动和停止：起步、低速巡航、减速，10 秒以后停车保持。
3. 正常行驶平衡：起步后进入持续巡航，直到 20 秒回合结束。
4. 行驶中加减速：先保持低速，再加速到较高速，随后减速到低速并保持。
5. 多次启动和停止：同一回合完成两次起步与停车，16 秒以后保持。

每回合最长 20 秒；策略跌倒或身体触地就提前结束。退出码 `2` 表示有物理失败或查看窗口被关闭，**不是安装失败**。结果和原始轨迹仍会保存。

结果包括实际回合时长、速度误差、末速度、最后半秒最大前向速度、保持漂移。未运行到保持阶段时，对应指标为 `null`，不能把它记成零漂移。`final_half_second_max_speed_mps` 只在回合末尾指令接近零时具有停车判据意义；持续巡航和行驶中加减速任务不应要求该指标为零。`hold_drift_m` 仅在仿真时间达到 16 秒且前进/偏航指令均接近零时记录，因此持续运动任务通常不产生该指标。

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

续训继承检查点的环境、任务类和PPO设置，输出及TensorBoard写入新目录。配置与检查点中的契约指纹必须同时匹配；模型/控制/观测实现变化、动作模式不同、旧v0.1检查点会被拒绝。不要删除校验或仅凭125/6维度强行加载。保存了模型和优化器，但物理回合重新开始，不承诺随机数和物理帧级精确恢复。`--steps`是新增步数，允许更改`--n-envs`；续训不接受新的`--config`。

初期可把速度上限降为0.2 m/s：

```bash
diablo train --max-speed 0.2 --steps 200000 --n-envs 4 --output runs/foundation-low-speed
```

`--noise` 只加入小幅陀螺仪和编码器速度噪声，不代表完整动力学随机化或 sim2real 验证。默认速度上限0.5 m/s，可以通过配置自行改变；改变速度范围并不保证策略会完成任务。复杂地形、跳跃和传感器扩展应先开发对应环境与评测。

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

## 自定义任务、奖励和算法

先保留原始基线，再建立新的实验目录。两种动作模式不可混用，详见[映射公式](model-interface.md#电机零位和动作)。

### 通过JSON调环境和PPO

```bash
diablo train --config examples/torque.json --steps 2048 --output runs/my-torque
```

`environment`对应`Config`：回合时长、最高指令速度、观测噪声、`action_mode`（`position_velocity`或`torque`）。`ppo`传入SB3 PPO，可改`n_steps`、`batch_size`、`n_epochs`、学习率、网络宽度等。`--max-speed`/`--noise`会覆盖JSON中相应选项。种子、设备、并行环境数用CLI设置。最终有效配置写入`resolved.json`。

直接力矩模式不含站立PD和支撑前馈，零动作不会自动平衡；这是算法可选择的控制接口，不是已调好的策略。

### 替换任务和奖励

修改示例`examples/custom_task.py`中的`MyEnv`，或在自己的模块继承`DiabloEnv`：

- `get_command(sim_seconds)`返回机体系`[前进m/s, 偏航rad/s, 高度m]`。
- `reward_for_transition(transition, action, previous_action)`返回有限标量。`transition`提供该动作对应的指令、机体速度、陀螺、竖直度、高度误差、最后子步力矩、终止标记；这些真值只供训练监督。
- 保持基类`reset/step`的传感观测、历史更新和故障处理。需要改变物理模型或输入输出时，更新契约与检查，再开始新训练。

```bash
export PYTHONPATH="$PWD/examples${PYTHONPATH:+:$PYTHONPATH}"
diablo train --config examples/custom.json --steps 2048 --output runs/my-custom
diablo evaluate --model runs/my-custom/policy.zip --output runs/my-custom-eval
diablo train --resume runs/my-custom/policy.zip --steps 2048 --output runs/my-custom-resume
```

`environment_class`为可导入的`模块:类名`。评估、续训和导出自动使用保存的任务类；该模块须始终可导入。指纹覆盖其整个模块文件，同文件常量/辅助函数变动也会拒绝旧策略。其他依赖文件需你自行版本管理。默认四任务报告中，采用自定义指令的语义以你的实现为准。

### 使用其他算法

`DiabloEnv`是普通Gymnasium环境，可直接使用任意兼容算法：

```python
from diablo_training.env import Config, DiabloEnv

env = DiabloEnv(Config(action_mode="torque"))
observation, info = env.reset(seed=42)
try:
    for _ in range(100):
        # 将这里替换成你的策略及更新逻辑
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
finally:
    env.close()
```

CLI的保存/续训/导出当前针对SB3 PPO；其他算法需用其自己的训练器、检查点和导出器，保留同样的观测/动作映射，不能把任意`.zip`交给PPO加载。

### 一次验证完整流程

```bash
bash scripts/check_training.sh
```

在新目录集中执行10项检查、doctor、smoke、2048步训练、512步续训、首回合评估、真实轨迹导出。脚本接受评估物理失败退出码2，仍保留失败报告；其他错误立即失败。它还确认参数更新、累计步数增加、新TensorBoard日志。实际运动是否合格要看评估证据，不能用脚本通过替代。

### wheel和资源目录

`python -m pip install .`也支持非editable安装。离开源码仓库后，`diablo fetch`默认将资源放在`~/.cache/embodied-execution/third_party`；设置`DIABLO_ASSET_DIR=/你的资源根目录`可指定位置（根目录下面是`third_party/`）。默认实验写入执行命令时的工作目录`runs/`。源码安装的资源则默认在仓库内。

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
| `contract is incompatible` | 使用训练时的相同代码/模型/MuJoCo，或开始新实验；v0.1策略不能复用 |
| 自定义类导入失败 | 续训、评估、导出时同样配置该模块的`PYTHONPATH` |

## 9. 六人团队如何协作

| 负责人 | 首轮任务 |
|---|---|
| 集成 | 固定模型接口和完整验收命令，处理主阻塞 |
| 模型 | 对照公开机构与后续实物资料，核对零位、质量惯量和驱动参数 |
| 环境 | 维护 `env.py` 的课程、奖励和失败定义 |
| 训练 | 使用同一基线训练，管理预算和检查点 |
| 部署接口 | 核对观测时序与导出契约，先做离线推理 |
| 评估 | 保存首回合失败，独立比较速度、制动和保持表现 |

所有人共享一套模型和接口。先完成基础运动，再讨论复杂地形和跳跃。
