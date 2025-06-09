import numpy as np
import gymnasium as gym
from gymnasium import spaces
import MeleeServer

class MeleeEnv(gym.Env):
    def __init__(self, server_id):
        self.server_id = server_id

        self.observation_space = spaces.Dict({
            
        })