import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import gym_env.register_env  # 确保已经注册

env = gym.make(
    "PandaReach-v0"
)

obs, info = env.reset()

for i in range(10):

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        i,
        obs,
        reward,
        terminated,
        truncated
    )

    if terminated or truncated:
        obs, info = env.reset()