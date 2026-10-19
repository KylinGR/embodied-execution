# DIABLO 训练资源

核对日期：2026-10-19。以下区分物理资产、训练代码、SDK和策略权重。

| 资源 | 用途 | 本项目的处理 |
|---|---|---|
| [厂家开发者入口](https://github.com/DDTRobot) | 官方维护与推荐项目分类 | 查找正式接口；推荐项目不等于官方维护 |
| [DIABLO强化学习示例](https://github.com/DDTRobot/Col_DIABLO_Issac_RL) | Isaac Gym代码、URDF、网格、动作映射参考 | 只拉取固定提交的模型资产；不安装Isaac Gym |
| [原作者项目](https://github.com/L-SY/wheeled_bipedal_gym) | 训练示例原始来源 | 保留作者与来源 |
| [Wheel-Legged-Gym](https://github.com/clearlab-sustech/Wheel-Legged-Gym) | 上游算法/环境来源 | 参考资料，不作为运行依赖 |
| [官方Webots环境](https://github.com/DDTRobot/diablo-sim-env) | 闭环连杆、物理模型、仿真场景参考 | 供机械核对；其README说明没有提供控制器 |
| [ROS2 SDK](https://github.com/DDTRobot/diablo_ros2) | 运动模式、传感器与六电机状态 | 参考资料，本项目不连接ROS或电机 |
| [ROS1 SDK](https://github.com/DDTRobot/diablo_sdk) | 机器人控制与通信 | 参考资料 |
| [MuJoCo建模文档](https://mujoco.readthedocs.io/en/stable/modeling.html) | URDF/MJCF、接触、执行器 | 本项目物理平台 |
| [Gymnasium环境接口](https://gymnasium.farama.org/introduction/create_custom_env/) | reset/step/终止语义 | 本项目环境接口 |
| [Stable-Baselines3 PPO](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) | PPO、向量环境、保存重载 | 本项目使用其成熟实现，不重写PPO |

## 当前能下载哪些策略？

检查到的DIABLO训练示例主分支没有`.pt/.onnx`预训练权重，[Release为空](https://github.com/DDTRobot/Col_DIABLO_Issac_RL/releases)。本项目也不将随机初始化或短训练检查点标为成熟策略。

官方SDK提供`jump_mode`等模式接口，但这不等于开放跳跃网络权重、训练配方或指定落点控制。DIABLO关节级接口的可用性需要向厂家技术支持确认。

## 为什么不直接运行 `diablo_vmc`？

该上游实验任务的观测拼接实际为27维，而配置继承25维；其动作语义也与普通关节位置任务不同。这里只记录该源码问题，不把未经验证的入口暴露给新手。

本项目从MuJoCo环境开始提供独立、已明确的125维历史输入与6维动作契约，详见[模型接口](model-interface.md)。
