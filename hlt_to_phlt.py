import numpy as np
from replay_parser import localize_matrix, load_replay, replay_to_enc_obs_n_stuff, enc_obs_to_obs
from baselines.acer.halite_env import HaliteEnv

import subprocess
import json
import time
import uuid
import io
import zstd
import os

DIRECTORY = "replays/"
file_list = os.listdir(DIRECTORY)

for replay_file_num, replay_file_name in enumerate(file_list):
    if not (replay_file_name[-4:] == ".hlt"):
        continue

    enc_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks = (None,)*6
    
    for player in range(2):
        # player = np.random.randint(0, num_players-1)

        replay = load_replay(DIRECTORY + replay_file_name, player, False)
        env = HaliteEnv()
        gamma = 0.99
        eo, a, r, m, d, ma = replay_to_enc_obs_n_stuff(replay, env, gamma=gamma)

        if enc_obs is None:
            enc_obs = eo
            mb_actions = a
            mb_rewards = r
            mb_mus = m
            mb_dones = d
            # mb_masks = ma
        else:
            enc_obs = np.concatenate((enc_obs, eo), axis=0)
            mb_actions = np.concatenate((mb_actions, a), axis=0)
            mb_rewards = np.concatenate((mb_rewards, r), axis=0)
            mb_mus = np.concatenate((mb_mus, m), axis=0)
            mb_dones = np.concatenate((mb_dones, d), axis=0)
            # np.concatenate((mb_masks, ma), axis=0)

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

    #return enc_obs, mb_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks
    # actuall, np.save them instead

##    cut_point = mb_dones.nonzero()[0][int(len(mb_dones.nonzero()[0])/2)]
    chunk_size = 8000
    num_chunks = 1 + len(enc_obs)//chunk_size
    done_indices = mb_dones.nonzero()[0]
    cutoffs = [-1]
    for index in range(1, num_chunks+ 1):
        cutoffs.append(max(filter(lambda y: y<chunk_size*index, done_indices)))
    for cutoff_index in range(1, len(cutoffs)):
        start = cutoffs[cutoff_index - 1] + 1
        end = cutoffs[cutoff_index] +1 
        with open("sync/replays/"+str(int(time.time()*1e9)) + "_" + str(uuid.uuid4())+"_"+str(cutoff_index)+".phlt", "wb+") as f:
            g = io.BytesIO()
            np.savez(g, enc_obs[start:end], mb_actions[start:end],
                     mb_rewards[start:end], mb_mus[start:end],
                     mb_dones[start:end])
            g.seek(0)
            f.write(zstd.compress(g.read()))
        

    print("Done with", replay_file_num)

