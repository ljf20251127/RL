import numpy as np
from typing import Optional
import gymnasium as gym
from gym_env.MyMujoco import MyMujoco
from gym_env.MyPink import MyPink

class PandaReach(gym.Env):

    def __init__(self):
        # 目标位置和当前位置的差
        self.observation_space = gym.spaces.Box(
            low = np.array([-1,-1,-1], dtype = np.float32),
            high = np.array([1,1,1], dtype = np.float32),
            dtype = np.float32
        )
        # dx dy dz
        self.action_space = gym.spaces.Box(
            low = np.array([-0.004,-0.004,-0.004], dtype = np.float32),
            high = np.array([0.004,0.004,0.004], dtype = np.float32),
            dtype = np.float32
        )
        self.mujoco = MyMujoco(
            "robot_models/franka_emika_panda/scene.xml"
        )
        self.pink = MyPink(
            "robot_models/panda.urdf",
            "panda_hand_tcp",
            self.mujoco.origin_q
        )
        self.goal_low = np.array([0.4,-0.2,0.4], dtype=np.float32)
        self.goal_high = np.array([0.8,0.2,0.75], dtype=np.float32)
        self.goal= None

        self.space_min = np.array([0.3,-0.3,0.3], dtype=np.float32)
        self.space_max = np.array([0.9,0.3,0.85], dtype=np.float32)

        self.max_step = 100
        self.step_count = 0
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        self.step_count = 0
        super().reset(seed=seed)
        self.mujoco.reset()
        self.pink.reset()
        self.goal = self.np_random.uniform(
            self.goal_low,
            self.goal_high
        )
        observation = self.get_obs()
        info = {}
        return observation, info

    def get_obs(self):
        pos = self.mujoco.get_pos()
        obs = self.goal - pos
        return obs.astype(np.float32)
    
    # action参数满足action_space
    def step(self, action):
        terminated = False
        target_pos = self.mujoco.get_pos() + np.array(action)
        # 管理target_q是否在合理范围内，否则给予负reward
        target_pos = np.clip(
            target_pos,
            self.space_min,
            self.space_max
        )

        target_q, ik_success = self.pink.solve(target_pos)
        if not ik_success:
            terminated = True

        self.mujoco.step(target_q)
        observation = self.get_obs()
        dist = np.linalg.norm(observation)

        reward = -dist
        success = dist < 0.01
        if success:
            terminated = True

        self.step_count += 1
        truncated = (self.step_count >= self.max_step)

        info = {}
        return observation, reward, terminated, truncated, info


