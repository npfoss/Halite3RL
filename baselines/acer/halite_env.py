import gym.spaces as spaces
import numpy as np

class HaliteEnv:
    def __init__(self):
        self.num_envs = 1

        lows = np.zeros(7)
        highs = np.array([np.inf, # halite
                    1, # friendly ship indicator
                    np.inf, # friendly ship cargo
                    1, # friendly dropoffs
                    1, # enemy ship indicator
                    np.inf, # enemy ship cargo
                    1, # enemy dropoff
                    ])
        # self.observation_space = spaces.Box(lows, highs, dtype=np.int)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(32, 32, 7), dtype=np.int)
        # self.observation_space = spaces.Box(lows, highs, shape=(7, 32, 32), dtype=np.int)

        self.action_space = spaces.Discrete(7) # North, South, East, West, Still, BUILD_DROPOFF, move towards home (as numbers 0-6)
        self.action_to_num = {
                'n': 0,
                's': 1,
                'e': 2,
                'w': 3,
                'o': 4,
                'c': 5,
                'h': 6,
            }

        self.nstack = 7 # this is the number of timesteps to stack into one "observation"
                        # (the Atari models all take the last 4 frames (greyscale) as input to the network)
                        # NOT IMPLEMENTED for > 1

        

