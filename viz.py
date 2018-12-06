# convenient imports for debugging
import numpy as np
from baselines.acer.halite_env import HaliteEnv
import subprocess
from os import listdir
from os.path import isfile, join
from random import sample
import time
import zstd
import io
from baselines.acer.halite_env import HaliteEnv
from replay_parser import localize_matrix, load_replay, replay_to_enc_obs_n_stuff, enc_obs_to_obs



def print_obs(m):
    c = lambda val: '.' if val == 0 else val
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            print(c(m[i,j]), end='')
        print()


# better but not done:
# def print_obs(enc_ob):
#     c = lambda val, char: '.' if val == 0 else char
#     for i in range(enc_ob.shape[0]):
#         for j in range(enc_ob.shape[1]):
#             print(c(enc_ob[i,j,1], 'A')+c(enc_ob[i,j,, end='')
#         print()


''' for reading in .hlt files
env = HaliteEnv()
replay = load_replay(replay_file_name, '0')
eo, a, r, m, d, ma = replay_to_enc_obs_n_stuff(replay, env, gamma=.99)

'''

''' for reading in .phlt files
path = './sync/replays'
replay_filenames = [f for f in listdir(path) if isfile(join(path, f)) and '.phlt' in f]
filename = replay_filenames[8]

with open(join(path, filename), 'rb') as f:
    g = io.BytesIO()
    g.write(zstd.uncompress(f.read()))
    g.seek(0)
    data = np.load(g)
    enc_obs, actions, rewards, mus, dones = (data[i] for i in data)

'''