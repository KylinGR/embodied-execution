# PPO 复现与单因素奖励对照

先按 `tutorial.md` 安装依赖和固定版本资源，运行：

```bash
bash scripts/check_training.sh
```

它验证测试、模型、随机动作、训练、续训、首回合评估和导出的一致性。
短验收通过只能说明流程能运行，不能说明机器人已学会站立。

## WSL 注意事项

上游 CAD 资源有仅大小写不同的 `base_link.STL` 和 `base_link.stl`。
将资源放在 WSL 的 Linux 文件系统，避免放入默认不区分大小写的 Windows 盘：

```bash
export DIABLO_ASSET_DIR="$HOME/.local/share/embodied-execution"
diablo fetch
diablo doctor
```

Python 虚拟环境也建议放在 Linux 文件系统，避免跨文件系统复制依赖的额外开销。
若已有不完整资源目录，应先检查其状态；`fetch` 不会自动修复已有目录。
项目为 shell 脚本指定 LF 换行，避免 Windows Git 的 CRLF 转换使 Bash 失败。

## 奖励改动

`examples/upright_reward.py` 继承默认环境，只将奖励改为：

```text
reward = default_reward - 2 × (1 - upright)
```

`upright` 是机体竖直方向与世界竖直方向的对齐程度。假设是增加倾斜代价
可能改善早期平衡；这是待验证假设，不是已验证的性能优化。
模型、控制、指令、观测、结束条件和 PPO 参数保持原样。
任务类使用独立模块，因此检查点契约与默认环境区分；不要交叉加载。

## 成对实验

```bash
python scripts/compare_rewards.py --steps 16384 --seeds 42 43 44 \
  --output runs/reward-pilot
```

输出目录必须不存在。脚本分别从头训练默认和改动奖励，每组使用相同训练种子、
4 个环境、同样的预算；评估固定种子 314，并保留跌倒首回合，不重试挑选。
实验配置存入 `experiment.json`，每次训练另有完整的 `resolved.json`。
`comparison.csv` 持续写入各任务时长、结束标记、速度误差；日志、模型、轨迹均保留。

先比较平衡任务的回合时长，再检查其他任务是否真正到达起步、制动和保持阶段。
四个任务在起步前都可能表现相同，不能当作四份独立的平衡成功证据。
不同奖励的训练回报不可直接比较；应比较相同物理指标和相同预算下的行为。
三个训练种子的小预算试验只能提供探索性结果，不能证明收敛或统计显著性。

## 回放与续训

```bash
export PYTHONPATH="$PWD/examples${PYTHONPATH:+:$PYTHONPATH}"
MUJOCO_GL=egl diablo play --model runs/reward-pilot/upright-42/policy.zip \
  --video --output runs/upright-42-video
diablo train --resume runs/reward-pilot/upright-42/policy.zip \
  --steps 512 --output runs/upright-42-resume
```

评估退出码 2 表示物理失败或取消，视频和原始结果仍需检查。
`PYTHONPATH` 必须能找到训练时的任务模块；改模块后应重新训练。
