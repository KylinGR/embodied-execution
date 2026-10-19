# 来源与许可证

本仓库自行编写的适配、环境、命令和教程按根目录BSD-3-Clause许可证发布。该许可证不替代上游代码、机器人资产、软件依赖及商标的权利条款。

## DIABLO资源

上游：`https://github.com/DDTRobot/Col_DIABLO_Issac_RL.git`

固定提交：`590085a027fc9e20b1d7e976714cd195488b96b6`

作者来源：L-SY；其项目基于clearlab-sustech/Wheel-Legged-Gym、ETH/NVIDIA训练代码等。部分源文件有BSD-3-Clause声明，`setup.py`也声明BSD-3-Clause，但该提交没有顶层LICENSE或单独明确的模型资产授权文件。

因此本仓库**不复制发布其URDF、STL或原训练源码**。`diablo fetch`从原始仓库取得固定提交，保留其全部内容、版权头和Git记录。下载目录被Git忽略。使用者在商业应用或再分发模型资产前应确认权利人的具体许可；不能将本仓库的许可证套到这些资产上。

本项目的腿部PD/轮部速度控制及径向支撑前馈依据公开上游公式编写；模型来源、假设和变化见`docs/model-interface.md`。

## 软件依赖

- MuJoCo：Google DeepMind，Apache-2.0。
- Gymnasium：Farama Foundation，MIT。
- Stable-Baselines3：DLR-RM及贡献者，MIT。
- PyTorch：PyTorch贡献者，BSD风格许可证；以安装版本分发条款为准。
- NumPy、ONNX、ONNX Runtime、ImageIO及FFmpeg组件：各自独立的许可证。

依赖通过包管理器安装，不将其许可证声明改成项目自有声明。FFmpeg通过imageio-ffmpeg取得，视频编码分发遵循对应构建的条款。
