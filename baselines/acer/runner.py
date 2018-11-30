import numpy as np
from baselines.common.runners import AbstractEnvRunner
from baselines.common.vec_env.vec_frame_stack import VecFrameStack
from gym import spaces
import os
from replay_parser import localize_matrix, load_replay, replay_to_enc_obs_n_stuff

import subprocess
import json
import tensorflow as tf

# from IPython import embed

class HaliteRunner:

    def __init__(self, model, env):
        self.nact = 6
        self.nenv = 1
        self.nsteps = 501 # (max game len) *** MAY NEED TO VARY WITH GAME (?) buffer size affected by this...
        self.model = model
        self.batch_ob_shape = (self.nenv*(self.nsteps+1),) + env.observation_space.shape

        # self.obs_dtype = env.observation_space.dtype
        # self.ac_dtype = env.action_space.dtype
        self.nbatch = self.nenv * self.nsteps


    def run(self):

        #first, run a game.
        size = 32# np.random.choice([32, 40, 48, 56, 64])
        num_players = 2# if (np.random.random() < 0.5) else 4

        # pickle the model
        self.model.save("actor.ckpt")
        #with open("weights", "wb+") as f:
        #    pkl.dump(self.model._step, f)

        o = subprocess.check_output(['./acer_run.sh', str(size), str(num_players)])
        j = json.loads(o.decode("utf-8"))

        #next, parse the replay and take all the credit
        player = np.random.randint(0, num_players-1)
        replay_file_name = j['replay']

        replay = load_replay(replay_file_name, player)

        enc_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks = replay_to_enc_obs_n_stuff(replay)
        mb_obs = enc_obs_to_obs(enc_obs)


        # # enc_obs = np.split(self.obs, self.nstack, axis=3)  # so now list of obs steps
        # enc_obs = np.split(self.env.stackedobs, self.env.nstack, axis=-1)
        # mb_obs, mb_actions, mb_mus, mb_dones, mb_rewards = [], [], [], [], []
        # for _ in range(self.nsteps):
        #     actions, mus, states = self.model._step(self.obs, S=self.states, M=self.dones)
        #     mb_obs.append(np.copy(self.obs))
        #     mb_actions.append(actions)
        #     mb_mus.append(mus)
        #     mb_dones.append(self.dones)
        #     obs, rewards, dones, _ = self.env.step(actions)
        #     # states information for statefull models like LSTM
        #     self.states = states
        #     self.dones = dones
        #     self.obs = obs
        #     mb_rewards.append(rewards)
        #     enc_obs.append(obs[..., -self.nc:])
        # mb_obs.append(np.copy(self.obs))
        # mb_dones.append(self.dones)

        # enc_obs = np.asarray(enc_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        # mb_obs = np.asarray(mb_obs, dtype=self.obs_dtype).swapaxes(1, 0)
        # mb_actions = np.asarray(mb_actions, dtype=self.ac_dtype).swapaxes(1, 0)
        # mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
        # mb_mus = np.asarray(mb_mus, dtype=np.float32).swapaxes(1, 0)

        # mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)

        # # done is True on the last frame of a game
        # # (THIS IS THE IMPORTANT ONE TO GET RIGHT)
        # # mask is True on the first frame of a game (generated from dones)
        
        # mb_masks = mb_dones # Used for statefull models like LSTM's to mask state when done
        # mb_dones = mb_dones[:, 1:] # Used for calculating returns. The dones array is now aligned with rewards

        # # shapes are now [nenv, nsteps, []]
        # # When pulling from buffer, arrays will now be reshaped in place, preventing a deep copy.

        return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks

