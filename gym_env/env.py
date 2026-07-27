import numpy as np
import gymnasium as gym

class panda_reach(gym.env):

    def __init__(self):
        # 目标位置和当前位置的差
        self.observation_space = gym.spaces.Box(
            low = np.array([-1,-1,-1]),
            high = np.array([1,1,1]),
            dtype = np.float
        )
        # dx dy dz
        self.action_space = gym.spaces.Box(
            low = np.array([-0.01,-0.01,-0.01]),
            high = np.array([0.01,0.01,0.01]),
            dtype = np.float
        )
    # action参数满足action_space
    def step(self, action):
        
