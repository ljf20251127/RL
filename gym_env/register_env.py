from gymnasium.envs.registration import register

register(
    id="PandaReach-v0",
    entry_point="gym_env.env:PandaReach",
    max_episode_steps=300
)