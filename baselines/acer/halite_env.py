import gym.spaces as spaces
import numpy as np

class HaliteEnv:
    def __init__(self):
        self.num_envs = 1

        lows = np.zeros(7)
        highs = np.array([np.inf, # halite
                    1, # friendly dropoffs
                    1, # friendly ship indicator
                    np.inf, # friendly ship cargo
                    1, # enemy dropoff
                    1, # enemy ship indicator
                    np.inf, # enemy ship cargo
                    ])
        # self.observation_space = spaces.Box(lows, highs, dtype=np.int)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(64, 64, 1), dtype=np.int)
        # self.observation_space = spaces.Box(lows, highs, shape=(7, 64, 64), dtype=np.int)

        self.action_space = spaces.Discrete(6)

        

