# MuJoCo 并行与 GPU 使用

当前策略网络很小，而 MuJoCo 的接触求解、闭链约束和 `mj_step` 仍主要运行在 CPU。仅执行
`PPO(device="cuda")` 只会把网络前向、反向和优化器放到 GPU，不会改变物理采样瓶颈。原先的
`DummyVecEnv` 又在同一个 Python 进程内串行推进多个 MuJoCo 实例，因此 GPU 利用率低是正常现象。

本项目现在提供两级加速：

1. `SubprocVecEnv`：每个 MuJoCo 实例在独立进程运行，并行使用多个 CPU 核，PPO 在 CUDA 上更新。
2. GPU 原生物理后端：将训练迁移到 Isaac Gym/Isaac Lab 或 MJX-Warp。该路线需要单独验证闭链、
   接触、地形和观测契约，不能在 `embodied-execution` 的 MuJoCo 环境中只改 `device` 完成。

## 1. 检查 CUDA PyTorch

```bash
source .venv/bin/activate
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda build", torch.version.cuda)
print("available", torch.cuda.is_available())
print("devices", torch.cuda.device_count())
PY
```

如果服务器有 NVIDIA GPU 但 `available=False`，说明安装的是 CPU wheel。按主机驱动安装匹配的
CUDA wheel，下面以 CUDA 12.x 的官方索引为例，实际 URL 需与服务器驱动和 PyTorch 版本一致：

```bash
pip uninstall -y torch
export TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
pip install 'torch>=2.8,<3' --index-url "$TORCH_INDEX_URL"
```

`scripts/setup.sh` 有 GPU 时默认安装 PyPI wheel，并支持通过 `TORCH_INDEX_URL` 指定专用 wheel。

## 2. 先测物理采样吞吐

在任何长训前先比较 `dummy` 和 `subproc`：

```bash
diablo benchmark --vec-env dummy --n-envs 4 --steps 200
diablo benchmark --vec-env subproc --n-envs 8 --steps 200
diablo benchmark --vec-env subproc --n-envs 16 --steps 200
diablo benchmark --vec-env subproc --n-envs 32 --steps 200
```

选择 `transitions_per_second` 最高且 CPU 占用、内存和服务器负载可接受的一组。不要直接照搬
`n_envs=64`；每个子进程都会创建一个 MuJoCo 模型和数据对象。

共享服务器示例：

```bash
export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
taskset -c 8-23 nice -n 10 diablo train \
  --n-envs 16 \
  --vec-env subproc \
  --device cuda:0 \
  --cpu-threads 1 \
  --gpu-fraction 0.50 \
  --steps 5000000 \
  --output runs/flat-parallel
```

`--gpu-fraction` 限制 PyTorch allocator 的显存，不是 GPU 利用率限速。若服务器要求更严格的
算力配额，仍应使用 Slurm、cgroup、容器或管理员提供的 MPS 策略。

## 3. GPU 物理路线

### 方案 A：Isaac Gym

厂家公开的 `Col_DIABLO_Issac_RL` 已提供 PhysX GPU 并行环境和 PPO 参考。它适合本项目的大规模
崎岖地形训练，但需要：

- 单独安装 NVIDIA Isaac Gym Preview 4；
- 固定 Python/驱动版本；
- 将 `embodied-execution` 的机构、动作符号和观测做成一致性测试；
- 将默认 `plane` 切换为 `trimesh`，并加入本项目的地形课程。

### 方案 B：MJX-Warp

MuJoCo MJX-Warp 可以保留同一 MuJoCo 模型资产并在 GPU 上做批量物理，当前文档明确支持
`EQUALITY` 约束和接触，因此比纯 MJX-JAX 更适合带平行四边形闭环的刑天。需要安装：

```bash
pip install 'mujoco-mjx[warp]'
```

该路线要重点验证：

- 2 个 `CONNECT` 闭环残差；
- 轮与地面的接触数量上限 `naconmax`；
- 每世界约束上限 `njmax`；
- 批量 reset 是否使用 MJX 的 `Data.where`；
- CPU MuJoCo 与 GPU 后端在同一初始状态和动作下的短时轨迹偏差；
- 训练结果能否由 CPU MuJoCo 独立复验。

建议先实现 `gpu_training/mjx_smoke.py` 级别的批处理物理和吞吐测试，再接入 PPO。没有通过
同一动作序列、闭环残差、轮前进方向和接触稳定性对照前，不要直接用 GPU 后端替换正式训练环境。

## 4. 选择顺序

1. `dummy + cpu`：功能回归和单元测试；
2. `subproc + cuda`：当前项目最快可落地路径；
3. Isaac Gym 或 MJX-Warp：当 `subproc` 的采样吞吐成为瓶颈后迁移；
4. 任何 GPU 后端都必须保留 CPU MuJoCo 作为独立复验器。
