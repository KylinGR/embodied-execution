import torch
import math

class DiabloJumpEnv:
    def __init__(self):
        """
        初始化跳跃环境
        """
        print("Initializing Diablo Jump Environment...")
        
        # 初始化仿真中的基础状态变量（防止代码运行时找不到变量而报错）
        self.root_states = torch.zeros(1, 13)      # 机器人根状态（其中索引 2 代表 Z 轴高度）
        self.base_lin_acc = torch.zeros(1, 3)     # 基础线加速度（用于计算落地冲击）
        self.angular_vel = torch.zeros(1, 3)      # 角速度（其中索引 1 代表 Pitch 俯仰角速度）
        
    def reset(self):
        """
        重置环境（每次训练重新开始时调用）
        """
        # 返回初始观察值 (Observations)
        return torch.zeros(1, 48)  # 示例特征维度

    def step(self, actions):
        """
        执行一步动作，计算奖励并返回结果
        """
        # 1. 执行电机动作 (在这里控制关节)
        # 2. 模拟更新机器人状态...
        
        # 计算各项奖励
        reward = self._compute_rewards()
        
        done = False  # 判断是否摔倒或结束
        observation = torch.zeros(1, 48)
        return observation, reward, done, {}

    def _compute_rewards(self):
        """
        核心：奖励函数设计（引导机器人学会跳跃与缓冲）
        """
        total_reward = 0.0
        
        # 1. 获取当前状态数据
        base_height = self.root_states[:, 2]         # 基座高度 (Z轴)
        vertical_acc = self.base_lin_acc[:, 2]      # 垂直方向加速度/冲击力
        pitch_rate = self.angular_vel[:, 1]         # Pitch 轴角速度（俯仰旋转）
        
        # ----------------------------------------------------
        # 核心奖励与惩罚项
        # ----------------------------------------------------
        
        # 1. 起跳高度奖励：鼓励机器人的身体向上抬高（目标高度 0.3 米）
        target_height = 0.3
        jump_height_reward = base_height - target_height  
        
        # 2. 落地缓冲惩罚：防止硬着陆砸坏（垂直加速度太大就扣分）
        landing_shock_penalty = -torch.square(vertical_acc).mean()
        
        # 3. 姿态稳定惩罚：防止起跳和空中时发生侧翻（限制俯仰角晃动）
        orientation_penalty = -torch.square(pitch_rate).mean()
        
        # 累加各项奖励与权重
        total_reward += (
            1.0 * jump_height_reward + 
            0.5 * landing_shock_penalty + 
            0.3 * orientation_penalty
        )
        
        # 【修改这里】使用 .item() 将 PyTorch Tensor 转为 Python 纯数字浮点数
        return total_reward.item()