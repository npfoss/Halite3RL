import numpy as np
import subprocess
from os import listdir
from os.path import isfile, join
from random import sample
import time
import zstd
import io

from IPython import embed

class Buffer(object):
    # gets obs, actions, rewards, mu's, (states, masks), dones
    def __init__(self, env, nsteps, size=50, disk_size=250):
        self.nenv = env.num_envs
        self.nsteps = nsteps
        # self.nh, self.nw, self.nc = env.observation_space.shape
        self.obs_shape = env.observation_space.shape
        self.obs_dtype = env.observation_space.dtype
        self.ac_dtype = env.action_space.dtype
        self.nc = self.obs_shape[-1]
        self.nstack = env.nstack
        self.nc //= self.nstack
        self.nbatch = self.nenv * self.nsteps

        ''' The New Deal:

            two buffers: one on disk, in sync/, holds many parsed replays from the distributed actors.
                this buffer is large. does not fit in memory all at once.

            buffer in RAM: basically what was here before, but instead of doing an on-policy update,
                it pulls new replays from the actors and samples a subset of the replays on disk to
                serve as the buffer for a little while (until next "on-policy" update)

            works the same as before with in-memory buffer (mostly)
        '''

        self.size = size# // (self.nsteps)  # Each loc contains nenv * nsteps frames, thus total buffer is nenv * size frames
        self.disk_size = disk_size

        # Memory
        self.enc_obs = None
        self.actions = None
        self.rewards = None
        self.mus = None
        self.dones = None
        self.masks = None

        # Size indexes
        self.next_idx = 0
        self.num_in_buffer = 0
        self.current_proc = None

    def has_atleast(self, frames):
        # Frames per env, so total (nenv * frames) Frames needed
        # Each buffer loc has nenv * nsteps frames
        return self.num_in_buffer >= (frames // self.nsteps)

    def can_sample(self):
        return self.num_in_buffer > 0

    # Generate stacked frames
    def decode(self, enc_obs, dones):
        # enc_obs has shape [nenvs, nsteps + nstack, nh, nw, nc]
        # dones has shape [nenvs, nsteps]
        # returns stacked obs of shape [nenv, (nsteps + 1), nh, nw, nstack*nc]

        return enc_obs
        return _stack_obs(enc_obs, dones,
                          nsteps=self.nsteps)

    def put(self, enc_obs, actions, rewards, mus, dones, masks=None):
        # enc_obs [nenv, (nsteps + nstack), nh, nw, nc]
        # actions, rewards, dones [nenv, nsteps]
        # mus [nenv, nsteps, nact]

        """ NEW BUFFER stuff:

            enc_obs: long concatenated list of ship traces for a single game (?, 64, 64, 7)
            same for the other things

            still stored in rows so we can sample uniformly from games, then from that game

            also, see in init: "The New Deal"
        """

        if self.enc_obs is None:
            self.enc_obs = [None] * self.size
            self.actions = [None] * self.size
            self.rewards = [None] * self.size
            self.mus = [None] * self.size
            self.dones = [None] * self.size
            # self.masks = [None] * self.size
            # self.enc_obs = np.empty([self.size] + list(enc_obs.shape), dtype=self.obs_dtype)
            # self.actions = np.empty([self.size] + list(actions.shape), dtype=self.ac_dtype)
            # self.rewards = np.empty([self.size] + list(rewards.shape), dtype=np.float32)
            # self.mus = np.empty([self.size] + list(mus.shape), dtype=np.float32)
            # self.dones = np.empty([self.size] + list(dones.shape), dtype=np.bool)
            # self.masks = np.empty([self.size] + list(masks.shape), dtype=np.bool)

        assert tuple(enc_obs.shape[1:]) == (64, 64, 7), 'enc_obs wrong shape! {}'.format(enc_obs.shape)
        self.enc_obs[self.next_idx] = enc_obs
        assert tuple(actions.shape[1:]) == (), 'actions wrong shape! {}'.format(actions.shape)
        self.actions[self.next_idx] = actions
        assert tuple(rewards.shape[1:]) == (), 'rewards wrong shape! {}'.format(rewards.shape)
        self.rewards[self.next_idx] = rewards
        # assert tuple(mus.shape[1:]) == (64, 64, 7), 'mus wrong shape! {}'.format(mus.shape)
        self.mus[self.next_idx] = mus
        assert tuple(dones.shape[1:]) == (), 'dones wrong shape! {}'.format(dones.shape)
        self.dones[self.next_idx] = dones
        # self.masks[self.next_idx] = masks

        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

    def take(self, x, idx, where_to_sample, one_extra=False):
        nenv = self.nenv
        out = np.empty([nenv] + [self.nsteps + one_extra] + list(x[0].shape[1:]), dtype=x[0].dtype)
        for i in range(nenv):
            # annoying: sometimes have to wrap around. and add some from the beginning of x to the end of out
            out[i] = x[idx[i]].take(np.arange(where_to_sample[i], where_to_sample[i] + self.nsteps + one_extra), mode='wrap', axis=0)
        return out

    def get(self):
        # returns
        # obs [nenv, (nsteps + 1), nh, nw, nstack*nc]
        # actions, rewards, dones [nenv, nsteps]
        # mus [nenv, nsteps, nact]
        # ^ no longer true: see NEW BUFFER comment above

        nenv = self.nenv # remember, nenv is number of uncorrelated samples to take --Nate
        assert self.can_sample()

        # Sample exactly one id per env. If you sample across envs, then higher correlation in samples from same env.
        idx = np.random.randint(0, self.num_in_buffer, nenv)

        # get which self.nsteps long chunk to take from the chosen games
        #   it's not important that it uses dones, they're all the same length
        where_to_sample = np.array([np.random.randint(0, self.dones[i].shape[0]) for i in idx])

        take = lambda x, oe=False: self.take(x, idx, where_to_sample, one_extra=oe)  # for i in range(nenv)], axis = 0)
        dones = take(self.dones)
        enc_obs = take(self.enc_obs, oe=True)
        obs = self.decode(enc_obs, dones)
        actions = take(self.actions)
        rewards = take(self.rewards)
        mus = take(self.mus)
        # masks = take(self.masks)
        # masks = np.array([[False]*self.nsteps for i in range(nenvs)])
        masks = None
        return obs, actions, rewards, mus, dones, masks

    def update_buffers(self):
        # sample new replays from disk
        #   takes 0.13285534381866454 sec on average to load enc_obs 1434 long
        path = './replays'
        replay_filenames = [f for f in listdir(path) if isfile(join(path, f)) and '.phlt' in f]

        for filename in sample(replay_filenames, min(self.size, len(replay_filenames))):
            with open(join(path, filename), 'rb') as f:
                g = io.BytesIO()
                g.write(zstd.uncompress(f.read()))
                g.seek(0)
                data = np.load(g)
                enc_obs, actions, rewards, mus, dones = (data[i] for i in data)
                try:
                    self.put(enc_obs, actions, rewards, mus, dones)
                except:
                    print('failed in update_buffers')
                    embed()

        # now update disk last to avoid concurrency problems
        # well, first have to check if the last one is done: poll() checks if process is still running
        while self.current_proc is not None and self.current_proc.poll() is None:
            # not done yet!
            print("waiting on the last one to finish! darn. increase replay_ratio maybe?")
            # wait n seconds
            time.sleep(1)

        self.current_proc = subprocess.Popen(['sh', 'update_replays.sh'])

