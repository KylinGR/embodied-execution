# 平稳落地：冻结反馈 + PPO 残差

2026-10-22，在本地 RTX 2080 Ti 上训练 200704 步。当前同一模型通过两组独立验收：低落差 30/30、宽落差 30/30，每回合完整 20 秒；精确 2 cm、5 cm 各补测一回合也通过。验收要求双轮有效接触、无非轮支撑、末段连续稳定至少 3 秒、最大水平漂移 ≤0.2 m、末 5 秒位置跨度 ≤0.05 m。非轮支撑按 100 Hz 采样、法向力 >1 N 判断。

宽落差实际随机样本覆盖 2.0068–4.9315 cm，最大漂移 0.088998 m，最短末段连续稳定 19.78 s。完整统计、源码指纹和每回合指标见 [验收报告](../artifacts/landing-v3/wide/result.json)。两个端点补测见 [端点报告](../artifacts/landing-v3/endpoints/result.json)。这些是采样测试，不是连续区间内所有初始条件的保证。

这是传感器 LQR 反馈与有界 PPO 残差组成的系统。基础反馈本身也通过验收，因此不据此宣称 PPO 提升了性能或独立学会落地。范围限于当前 MuJoCo 模型、平地、已知直立释放、无偏仿真陀螺。高度是复位基础轮地间隙约 5 mm 之上的额外抬升量；未验证主动起跳、离台、越障、传感器偏置、随机姿态或实机。

## 保存的产物

- [模型与配置](../artifacts/landing-v3/train/)：policy.zip、resolved.json、训练报告及独立审查。
- [低落差报告](../artifacts/landing-v3/low/result.json)、[宽落差报告](../artifacts/landing-v3/wide/result.json)。
- [60 秒实际回放](../artifacts/landing-v3/wide/qualification.mp4)：三个高度各首个固定种子的完整 20 秒过程。
- [实验日志](landing-training-log-20261022.md)：失败诊断、旧评价锁存错误、控制器与 PPO 的贡献边界。

完整逐控制步轨迹保留于原执行机器的 runs/，不纳入 Git；以下命令可重新生成。日志和审查中 runs/ 路径是原始执行路径；发布副本对应 artifacts/landing-v3/。不复制上游机器人网格或模型资产，仍按 THIRD_PARTY.md 从固定上游获取。

## Linux / WSL 重放

按主教程安装并 `diablo fetch` 后，在仓库根目录运行。CUDA 只用于训练，验收在 CPU 上运行策略。不要改变冻结源码的字节/换行，否则契约会拒绝加载；.gitattributes 已为相关文件保留原始字节。

```bash
export PYTHONPATH="$PWD/examples${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python examples/qualify_landing.py \
  --model artifacts/landing-v3/train/policy.zip \
  --heights .025 .035 .045 --jitter .005 --seed-start 12000 \
  --video --output runs/landing-replay
```

输出目录必须不存在。重新训练：

```bash
diablo train --device cuda --steps 200000 --n-envs 4 --seed 42 \
  --config examples/stable_landing_ppo_v3.json --output runs/landing-train
```

冻结入口是 stable_landing_v3.py、stable_landing_fast_v3.py、stable_landing_ppo_v3.json 和 qualify_landing.py；check_stable_landing.py 验证接口、力矩、零残差一致性与新旧契约。其他 landing/search/identify/balance 示例保留失败实验和辨识过程，不能替代当前严格资格器。evaluate_landing.py 的历史指标有成功锁存，仅供诊断。

参考：[MuJoCo API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)、[SB3 训练与评估建议](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)、[离散 LQR](https://python-control.readthedocs.io/en/stable/generated/control.dlqr.html)。
