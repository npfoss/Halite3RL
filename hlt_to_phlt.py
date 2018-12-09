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

DIRECTORY = "expert_replays/"
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

