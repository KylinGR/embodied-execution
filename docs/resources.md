# 资源与来源

核对日期：2026-10-28。

| 资源 | 用途 | 本项目处理 |
|---|---|---|
| [用户提供的刑天说明书](https://flbook.com.cn/v/Il0Y7WuaLU?cckey=c124f578) | 产品与操作说明 | 页面启用人机验证；未自动绕过，参数以可访问的官网页和公开代码复核 |
| [本末刑天产品页](https://robotics.directdrive.com/products/xingtian-robot/1653337674873306) | 官方质量、速度、负载和整机版本 | 22.9 kg 用于质量归一化；尺寸版本差异单独记录 |
| [厂家 RL 示例](https://github.com/DDTRobot/Col_DIABLO_Issac_RL) | CAD URDF、网格、惯量 | 固定提交后只读取所需资产 |
| [官方 Webots 环境](https://github.com/DDTRobot/diablo-sim-env) | 髋膝串联、回程支链、闭环 | 固定提交后用于机构校核 |
| [DIABLO 论文](https://arxiv.org/html/2407.21500v1) | 机构及 M1502D 电机参考 | 用于 17 Nm / 23.038 rad/s 等公开参数 |
| [MJLab](https://github.com/mujocolab/mjlab) | GPU 物理训练、管理器式任务、RSL-RL | 固定 `1.6.0` |
| [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) | GPU 批量物理 | 通过 MJLab 的 `~=3.11.0` 依赖 |
| [MuJoCo](https://mujoco.readthedocs.io/) | CPU 模型编译和机构校核 | 固定 `3.11.0` |
| [RSL-RL](https://github.com/leggedrobotics/rsl_rl) | PPO Actor/Critic | 使用 MJLab 锁定的 `5.4.2` |

## 当前不提供的内容

- 没有可下载的刑天预训练策略；
- 没有实机电机零位、驱动固件、热模型和通信延迟标定；
- 没有宣称 491 mm 官网版本与课程 379.58 mm CAD 版本完全一致；
- 没有将论文样机或公开 CAD 参数当作已称重实机。

## 为什么只保留 GPU 训练路径

旧单环境训练让 MuJoCo 状态、奖励和采样围绕 Python 对象逐环境推进，无法让物理、观测、奖励和 PPO 采样一起运行在 MuJoCo Warp 批处理上。项目已统一使用 MJLab + RSL-RL；CPU MuJoCo 只做校核。

## 文档来源边界

用户提供的 flbook 说明书链接启用了交互式人机验证，未自动绕过。参数核对使用了本末机器人官网公开产品页和厂商公开代码/论文。若后续拿到说明书中的具体硬件版本参数，应新增带版本的模型配置，而不是覆盖现有契约。
