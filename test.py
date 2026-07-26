import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO.load("ppo_cartpole")

obs, info = env.reset()

while True:

    action, _ = model.predict(obs)

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        # obs, info = env.reset()
        break

env.close()