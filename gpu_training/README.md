# GPU 物理训练参考

本目录不替代当前 MuJoCo CPU 环境。它记录两条 GPU 原生物理路线，以及进入正式训练前必须通过的
模型一致性门槛。

## MJX-Warp 快速验证

MJX-Warp 直接读取当前 `make_model()` 生成的 MuJoCo 模型，支持 `EQUALITY` 约束和接触，适合先验证
刑天并联闭环能否在 GPU 上稳定批处理。

```bash
pip install 'mujoco-mjx[warp]'
python -m gpu_training.mjx_warp_benchmark \
  --impl warp --num-envs 256 --steps 20
```

如果出现接触或约束溢出，增大 `--naconmax`/`--njmax` 并监控显存。若只是吞吐验证，先用
`--num-envs 64/128/256/512` 做阶梯测试。

验收门：

- 所有世界 `qpos/qvel` 有限；
- 左右平行四边形闭环误差不超过 MuJoCo CPU 基线；
- 零动作下短时间质量中心下降趋势与 CPU MuJoCo 一致；
- 相同小动作下左右轮前进符号一致；
- 100 步以上无 NaN、无约束爆炸、无非预期自碰撞。

## Isaac Gym 训练

厂家公开工程：

```bash
git clone https://github.com/DDTRobot/Col_DIABLO_Issac_RL.git
cd Col_DIABLO_Issac_RL
```

该工程的默认配置仍是 `mesh_type = "plane"`。正式使用前至少完成：

1. 固定 Isaac Gym Preview 4、Python 和驱动版本；
2. 核对当前 `diablo_training` 的关节名、动作符号、默认角和轮速方向；
3. 将地形切换为 `trimesh`，使用 5 cm 水平分辨率、5 mm 垂直分辨率；
4. 添加连续崎岖、坡面、低障碍和混合路线；
5. 保留 `embodied-execution` 作为 CPU MuJoCo 独立回归；
6. 用同一个 125/6 导出契约检查最终策略。

## 选择建议

先用 `SubprocVecEnv + CUDA PPO` 完成平地能力，因为这条路径改动小、可直接测试。只有采样成为
明确瓶颈时，才迁移到 GPU 物理。不要同时更换物理引擎、奖励、地形和网络结构；否则无法判断
性能变化来自哪里。
