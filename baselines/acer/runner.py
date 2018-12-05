import numpy as np
from replay_parser import localize_matrix, load_replay, replay_to_enc_obs_n_stuff, enc_obs_to_obs
from baselines.acer.halite_env import HaliteEnv

import subprocess
import json

# from IPython import embed

class HaliteRunner:

    def __init__(self):
        with open("params.json") as f:
            params = json.load(f)

        self.env = HaliteEnv()
        self.nact = 6
        self.nenv = 1
        self.nsteps = params["nsteps"]
        self.batch_ob_shape = (self.nenv*(self.nsteps+1),) + env.observation_space.shape

        # self.obs_dtype = env.observation_space.dtype
        # self.ac_dtype = env.action_space.dtype
        self.nbatch = self.nenv * self.nsteps
        self.gamma = params["gamma"]


    def run(self):

        #run a game.
        size = np.random.choice([32, 40, 48, 56, 64])
        num_players = 2 if (np.random.random() < 0.5) else 4

        o = subprocess.check_output(['./acer_run.sh', str(size), str(num_players)])
        j = json.loads(o.decode("utf-8"))

        #next, parse the replay
        replay_file_name = j['replay']
        enc_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks = (None,)*6

        for player in range(num_players):
            # player = np.random.randint(0, num_players-1)

            replay = load_replay(replay_file_name, player)

            eo, a, r, m, d, ma = replay_to_enc_obs_n_stuff(replay, self.env, gamma=self.gamma)

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
        with open(str(int(time.time()*1e9)) + "_" + str(uuid.uuid4())+".phlt", "wb+") as f:
            np.save(f, np.array([enc_obs, mb_actions, mb_rewards, mb_mus, mb_dones]))

if __name__ == "__main__":
    with open("params.json") as f:
        runs = json.load(f)["actor_runs_per_upload"]
    runner = HaliteRunner()
    current_proc = None
    while True:
        for i in range(runs):
            runner.run()
        while current_proc is not None and current_proc.poll() is not None:
            # not done yet!
            print("waiting on previous upload")
            # wait n seconds
            time.sleep(1)
        current_proc = subprocess.Popen(['./actor_upload.sh'])

