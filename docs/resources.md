# 资源与来源

核对日期：2026-10-31。

| 资源 | 用途 | 本项目处理 |
|---|---|---|
| [用户提供的刑天说明书](https://flbook.com.cn/v/Il0Y7WuaLU?cckey=c124f578) | 产品与操作说明 | 页面启用人机验证；未自动绕过，参数以可访问的官网页和公开代码复核 |
| [本末刑天产品页](https://robotics.directdrive.com/products/xingtian-robot/1653337674873306) | 官方产品规格和版本差异 | 只作外部产品参考；不再用 22.9 kg 覆盖主分支 CAD 质量 |
| [厂家 RL 示例](https://github.com/DDTRobot/Col_DIABLO_Issac_RL) | CAD URDF、网格、惯量 | 固定提交后只读取所需资产 |
| [官方 Webots 环境](https://github.com/DDTRobot/diablo-sim-env) | 髋膝串联、回程支链、闭环 | 固定提交后用于机构校核 |
| [DIABLO 论文](https://arxiv.org/html/2407.21500v1) | 机构及 M1502D 电机参考 | 用于 17 Nm / 23.038 rad/s 等公开参数 |
| [DDT Robot Lab](https://github.com/cmjang/ddtrobot_lab) | 同厂 Tita 轮足 IsaacLab 奖励、命令课程和 RSL-RL PPO | 参考奖励量级与分阶段参数趋势，不复制模型或训练代码 |
| [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) | 开源轮腿机器人 PPO 基线 | 参考速度/转向跟踪、姿态和动作率权重量级 |
| [MJLab](https://github.com/mujocolab/mjlab) | GPU 物理训练、管理器式任务、RSL-RL | 固定 `1.6.0` |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | GPU 批量物理 | 通过 MJLab 的 `~=3.11.0` 依赖 |
| [MuJoCo](https://mujoco.readthedocs.io/) | CPU 模型编译和机构校核 | 固定 `3.11.0` |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | PPO Actor/Critic | 使用 MJLab 锁定的 `5.4.2` |

## 当前不提供的内容

- 没有可下载的刑天预训练策略；
- 没有实机电机零位、驱动固件、热模型和通信延迟标定；
- 没有宣称 491 mm 产品上部包络与当前 477.859 mm 健康行驶站姿完全一致；
- 没有将论文样机或公开 CAD 参数当作已称重实机。

## 为什么只保留 GPU 训练路径

旧单环境训练让 MuJoCo 状态、奖励和采样围绕 Python 对象逐环境推进，无法让物理、观测、奖励和 PPO 采样一起运行在 MuJoCo Warp 批处理上。项目已统一使用 MJLab + RSL-RL；CPU MuJoCo 只做校核。

## 文档来源边界

用户提供的 flbook 说明书链接启用了交互式人机验证，未自动绕过。参数核对使用了本末机器人官网公开产品页和厂商公开代码/论文。若后续拿到说明书中的具体硬件版本参数，应新增带版本的模型配置，而不是覆盖现有契约。
