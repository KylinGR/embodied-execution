import torch
from diablo_jump_env import DiabloJumpEnv

def main():
    print("=== 准备启动 DIABLO 跳跃训练测试 ===")
    
    # 1. 实例化我们刚才写好的跳跃环境
    env = DiabloJumpEnv()
    
    # 2. 重置环境，获取初始观察值
    obs = env.reset()
    print("环境重置成功，初始观察值维度:", obs.shape)
    
    # 3. 模拟一个简单的训练交互循环 (Training Loop)
    # 在真实的强化学习中，这里会接入 PPO 算法，让神经网络去学着输出最优动作
    for episode in range(3):  # 模拟 3 个训练回合 (Episode)
        print(f"\n--- 开始第 {episode + 1} 个训练回合 ---")
        obs = env.reset()
        
        total_episode_reward = 0.0
        
        for step in range(5):  # 每个回合运行 5 步
            # 模拟模型输出一个随机动作（后续这里会替换为 AI 算法输出）
            simulated_action = torch.randn(1, 12) 
            
            # 与环境进行交互：执行动作，获取下一状态、奖励、是否结束标志
            next_obs, reward, done, info = env.step(simulated_action)
            
            total_episode_reward += reward
            print(f"步骤 {step + 1}: 当前步获得奖励 (Reward) = {reward:.4f}")
            
            if done:
                break
                
        print(f"第 {episode + 1} 回合结束，总奖励: {total_episode_reward:.4f}")

    print("\n=== 恭喜！跳跃环境连通性测试通过！ ===")

if __name__ == "__main__":
    main()