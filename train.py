import gymnasium as gym
from stable_baselines3 import PPO

# 创建环境
env = gym.make("CartPole-v1")

# 创建PPO模型
model = PPO("MlpPolicy", env, verbose=1)

# 加载模型
# model = PPO.load("ppo_cartpole", env=env)

# 开始训练
model.learn(total_timesteps=40000)

# 保存模型
model.save("ppo_cartpole")