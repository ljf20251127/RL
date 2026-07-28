import gymnasium as gym
import numpy as np

import gym_env.register_env


env = gym.make("PandaReach-v0")


obs, info = env.reset()


print("initial obs:", obs)


for i in range(100):

    # obs = goal - current_pos
    direction = obs / (np.linalg.norm(obs) + 1e-8)

    # 每步移动4mm
    action = 0.004 * direction


    obs, reward, terminated, truncated, info = env.step(action)


    distance = np.linalg.norm(obs)


    print(
        "step:",
        i,
        "distance:",
        distance,
        "reward:",
        reward
    )


    if terminated or truncated:
        print("episode end")
        break


env.close()