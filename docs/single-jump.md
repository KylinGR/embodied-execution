# 平地单次原地小跳：主动蹬地、真实腾空、落地保持

2026-10-27，本地RTX2080Ti完成200704步新PPO训练，并使用同一冻结策略完成30/30独立仿真资格验收。每回合完整20秒，起跳前连续站立，再主动蹬地，完成一次约2厘米双轮真实腾空，落地后保持站立至结束。无摔倒或超过1N的非轮支撑。

| 指标 | 新PPO残差系统 | 零残差反馈基线 | 门槛 |
|---|---:|---:|---:|
| 完整20秒回合 | 30/30 | 30/30 | 30/30 |
| 双轮较低净空峰值 | 1.913–1.960 cm | 1.925–1.962 cm | 1.5–3 cm |
| 连续真实腾空 | 0.132–0.134 s | 0.132–0.134 s | >=0.060 s |
| 最短末段连续稳定 | 17.522 s | 17.520 s | >=3 s |
| 最大全程水平漂移 | 0.138915 m | 0.138657 m | <=0.2 m |
| 最大末5秒漂移 | 0.001280 m | 0.001384 m | <=0.05 m |

测量为500Hz物理子步。净空由双轮碰撞圆柱的世界朝向、半径、半宽精算；质心竖直上升使用整机子树世界速度，最低上升峰值0.634m/s，排除仅伸腿抬机身或初始自由下落。起跳事件需紧邻地面支持，且固定在1–3秒窗；起跳前至少0.5秒稳定双轮支撑。落地后双轮支持中断最长2ms，明确允许的上限为20ms；不声称每个瞬间绝对连续接触。

## 产物与复现

- [模型、解析配置、训练报告与日志](../artifacts/single-jump-v1/train/)。新模型148维，不能加载旧143维自由落地模型代替。
- [最终资格报告](../artifacts/single-jump-v1/qualification/result.json)、[60秒实际视频](../artifacts/single-jump-v1/qualification/qualification.mp4)：前三个独立回合各完整20秒，25fps、640×480。全部30回合由完整轨迹支撑。
- [零残差基线报告](../artifacts/single-jump-v1/baseline-result.json)、[契约冻结](../artifacts/single-jump-v1/frozen.json)、[源码快照及SHA清单](../artifacts/single-jump-v1/source-manifest.json)。快照保留实际执行字节与换行。
- [实验过程](single-jump-training-log-20261027.md)、[资格器独立审查](jump-metrics-review.md)；17项指标正反例、10项原物理接口测试和Gym环境检查均通过。
- 完整物理轨迹位于 `runs/single-jump-ppo-v1-qualification-20261027/trace.jsonl.gz` 与 `runs/single-jump-baseline-v1-full-20261027/trace.jsonl.gz`，每份含300000有效子步；包括qpos/qvel、实际电机力矩、残差/基础/实际动作、质心/接触/几何指标。t=0初态缺状态字段，独立审查按同seed地面零初速复位重建，随后所有子步状态完整。

Linux/WSL重放，在仓库根目录执行：

```bash
bash examples/replay_single_jump_v1.sh
```

手工重新训练：

```bash
export DIABLO_ASSET_DIR=/home/qin/.local/share/embodied-execution
snapshot="$PWD/artifacts/single-jump-v1/source_snapshot"
export PYTHONPATH="$snapshot/examples:$snapshot:$PWD/examples:$PWD${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home/qin/.venvs/embodied-execution-gpu/bin/python examples/run_single_jump_v1.py train \
  --device cuda --steps 200000 --n-envs 4 --seed 43 \
  --config examples/single_jump_ppo_v1.json --output runs/single-jump-new-training
```

输出目录必须不存在。运行依赖为Torch2.8.0+cu126、SB3 2.9.0、MuJoCo3.8.1；CUDA实际运算已验证，MuJoCo物理在CPU运行。不得转换冻结文件换行，模型会按源码字节契约拒绝不匹配的执行环境。

## 独立最终审查

[完整审查JSON](../artifacts/single-jump-v1/train/independent_audit.json)从30回合原始轨迹重算300000个物理子步，全部指标与资格报告一致。每子步基础动作＋PPO残差的公式误差为0，实际力矩与原速度包络解码误差为0；峰值实际力矩低于17Nm。初始/最终参数指纹匹配，actor7/7、critic6/6张量更新；残差造成非零实际力矩变化。控制与actor未读取评估真值，记录器开关对同seed轨迹、观察与奖励差异为0。初态外力为0且源码没有施加外力或状态重置；轨迹未直接记录全程外力，故外力因果证据来自冻结源码及复位检查。

## 能力与归因边界

系统为搜索得到的主动脉冲、传感LQR反馈与经过真实训练的PPO有界力矩残差。actor和critic参数均更新，不能据奖励上涨推断物理成功；物理成功来自上述完整资格。奖励主要保持姿态和位置，没有从零搜索起跳事件。零残差基线也30/30通过，因此不宣称PPO学会独立跳跃或带来性能提升。

最终种子21000–21029预先冻结，未用于参数搜索，也未重试筛选；基线种子19000–19029。随机性仅为两项脉冲命令各±.001扰动，初始状态仍为已知直立、轮底贴地、零初速；无偏仿真陀螺。范围仅限当前模型和平地单跳，不覆盖姿态扰动、地形、连续跳、障碍、实机或多训练种子鲁棒性。17Nm与原力矩速度约束保留；500Hz记录不构成硬件冲击认证。

本轮模型、配置、视频与审查作为独立单跳实验包发布；完整大体积子步轨迹保留在原执行机器的runs/，可通过重放命令重新生成。旧自由落地证据和源码均保留；原HEAD关键源码字节另备份于 `runs/landing-v3-head-source-backup-20261027`。

## 发布复现补充（2026-10-29）

新训练执行时，基础模块的换行与上一轮Git提交不同。源码指纹覆盖字节，不能把“仅换行”当作契约相同。为同时保留旧落地模型和新单跳模型的可复现性，仓库根目录三个基础模块恢复为两轮模型共同要求的实际冻结字节（只修复旧Git发布的LF/CRLF不一致，不改任何物理或控制逻辑）；单跳脚本优先加载`artifacts/single-jump-v1/source_snapshot`中的完整Python包及任务源码。快照是本仓库自写实现，不包含上游URDF/STL。新模型验证使用原训练契约，不做强制兼容、不改权重或资格阈值。

`examples/single_jump_v1.py`等根目录任务文件用于阅读；带已发布检查点重放时，使用上述快照入口。源码快照的SHA清单与模型均在产物目录内。已有最终资格与审查报告记录原始执行路径；这些路径是证据出处，不要求新机器存在相同绝对路径。

干净检出验证发现：旧提交`58ee0d2`中的`diablo_training/env.py`、`model.py`、`control.py`使用LF，而旧落地检查点和新单跳检查点记录的都是当前保留的CRLF字节。旧提交干净检出加载旧模型会被契约正确拒绝。本次发布将这三个文件按已记录SHA恢复为CRLF，`.gitattributes`保留`-text`；这是发布格式纠错，不是重新训练、放松契约或更改仿真。新单跳原30回合资格和模型均保持不变；发布前在干净导出树中对旧落地模型（seed13000，2cm）及新单跳模型（seed21000）各完成一个20秒严格回放，均通过且契约匹配；补测不替换原30回合资格。[发布复验记录](../artifacts/single-jump-v1/publication-verification-20261029.json)。
