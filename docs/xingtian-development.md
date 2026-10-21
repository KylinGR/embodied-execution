# 在 embodied-execution 中开发刑天训练能力

团队每名成员都已经拥有仓库写权限，因此直接使用唯一中央仓库。所有人从中央仓库建立短分支，
通过 PR 合并；不使用个人 Fork。短训练失败证据和长训结果不提交 Git。

## 1. 克隆中央仓库

直接克隆有写权限的中央仓库：

```bash
git clone https://github.com/KylinGR/embodied-execution.git
cd embodied-execution
git fetch origin
git switch main
git pull --ff-only
```

首次建立基础组集成分支：

```bash
git switch -c base-locomotion-integration main
git push -u origin base-locomotion-integration
```

之后基础组成员从中央仓库建立短分支。`main` 必须开启分支保护，即使有写权限也禁止直接 push。

### VS Code Remote-SSH

1. 在 Remote-SSH 窗口打开远程 Linux 仓库目录；
2. 所有克隆、补丁、安装和训练命令都在 VS Code 远程终端执行；
3. 将补丁上传到远程文件系统后再执行 `git apply`；
4. 选择 `${workspaceFolder}/.venv/bin/python` 作为 Python 解释器；
5. 从 `Terminal → Run Task` 运行 `XINGTIAN:` 任务；
6. 需要曲线时启动 TensorBoard `--port 6006`，由 VS Code 自动转发端口。

不要同时在 Windows 和 Linux 编辑同一个工作树，避免 CRLF、符号链接和大小写差异。

## 2. 创建分组分支

```bash
git switch -c base/<member>/<topic> origin/base-locomotion-integration
git switch -c shared/<topic> origin/main
```

分支建议：

- `base-locomotion-integration`：基础运动组集成分支，仅基础运动 2 人负责。
- `base/<member>/<topic>`：基础运动短分支。
- `shared/<topic>`：观测、动作、机器人和并行训练设施改动。
- `feature/xingtian-eval`：独立评测与回归。
- `experiment/<stage>-<date>`：只放配置和结果摘要，不放大型权重。

不要把“GPU 并行”“奖励重构”“崎岖地形”和“网络结构变化”放进同一个不可审阅提交。

## 3. 安装与资产

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 如果自动选择 CUDA wheel失败：
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
bash scripts/setup.sh

diablo fetch
diablo doctor
python -m unittest discover -s tests -v
```

`TORCH_INDEX_URL` 必须与服务器驱动和 PyTorch 版本匹配。不要把第三方资产提交到主仓库：
`third_party/`、`.venv/`、`runs/` 均已被 `.gitignore` 排除。

## 4. 先测吞吐，再启动长训

```bash
diablo benchmark --vec-env dummy --n-envs 4 --steps 200
diablo benchmark --vec-env subproc --n-envs 8 --steps 200
diablo benchmark --vec-env subproc --n-envs 16 --steps 200
diablo benchmark --vec-env subproc --n-envs 32 --steps 200
```

用相同主机负载比较 `transitions_per_second`。训练命令示例：

```bash
CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
taskset -c 8-23 \
nice -n 10 \
diablo train \
  --device cuda:0 \
  --vec-env subproc \
  --n-envs 16 \
  --cpu-threads 1 \
  --gpu-fraction 0.5 \
  --steps 2000000 \
  --output runs/flat-a1
```

`CUDA_VISIBLE_DEVICES=1` 后，进程内的 `cuda:0` 是物理 1 号卡。

## 5. 开发顺序

### 提交 1：性能和工具链

- `SubprocVecEnv`；
- CUDA/CPU 自动选择；
- 吞吐基准；
- GPU wheel 安装提示；
- 单元测试。

验收：

```bash
python -m unittest tests.test_parallel -v
diablo train --device auto --vec-env dummy --n-envs 2 \
  --steps 2048 --output runs/integration-dummy
```

### 提交 2：任务契约

- 固定 25×5 可部署 actor 输入；
- 固定 6 维动作；
- 命令采样、终止和奖励分项；
- contract 哈希和 resolved 配置。

验收：

- `diablo doctor`；
- 固定 seed 两次 reset 一致；
- 动作符号、轮前进方向和偏航方向通过。

### 提交 3：地形与课程

- 平地、连续崎岖、坡面、小障碍；
- 地形等级与命令课程；
- 未见过 seed 的评测集；
- 失败轨迹和视频。

### 提交 4：评测与辅助技能

- 平衡、前进/后退、转向、启停；
- 35° 坡；
- 10～60 mm 障碍；
- 单轮抬升矩阵；
- ONNX/TorchScript 导出。

每提交一次都运行：

```bash
python -m unittest discover -s tests -v
git diff --check
```

## 6. 分组协作约定

- 基础运动固定由 2 人负责，二人都能独立启动训练、恢复训练、运行评测和定位失败；
- 基础运动每份实验结果必须由另一名基础组成员独立重载并评测；
- 训练操作员在基础组内轮换，避免只有一人掌握基础训练命令；
- 基础-A 主责算法时，基础-B 负责环境和评测复核；基础-B 修改奖励或地形时，基础-A 复核；
- 基础运动的接口、奖励、地形和 checkpoint 兼容性由这 2 人共同签字；
- 本流程不创建跳跃分支，也不管理跳跃训练、奖励、评测或 checkpoint；
- 基础组共享的模型、控制、观测和动作接口改动由基础组 2 人共同评审；
- 每周冻结一次可回归分支，长训从冻结点启动。

## 7. 提交与推送

```bash
git add diablo_training tests docs gpu_training
git diff --cached
git commit -m "feat(xingtian): add GPU-aware parallel training"
git push -u origin HEAD
```

PR 必须包含：

- 改动目的和实验环境；
- 改动类型：基础运动或共享接口；
- 是否改变观测、动作、奖励或模型契约；
- 对应单元测试；
- 短训前后的吞吐数据；
- 至少一个固定 seed 的评测结果；
- 未验证内容和已知风险。

大型 `.zip`、`.pt`、`.onnx`、视频和 `runs/` 不入库；只提交配置、指标摘要和复现命令。
