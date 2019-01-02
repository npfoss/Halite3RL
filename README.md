# Halite3RL
A reinforcement learning bot for the Halite III programming competition https://halite.io/

## install

```bash
wget https://github.com/npfoss/Halite3RL/blob/master/initializer.sh
# or if that doesn't work because the repo is still private:
# wget http://npfoss.com/initializer.sh
chmod +x initializer.sh
./initializer.sh
```

This initializer was written for Ubuntu 18.04 but should work on any Debian-based Linux distro.
If it doesn't work, try going through it and running your machine's equivalent of each command.

## running

to run the actors which generate data:
`python3 actor.py`

to run the learner (only one at a time please):
`python3 -m baselines.run --alg=acer --env=PongNoFrameskip-v4`
...it's fine.

to download the most recent weights, go to: [tracksingles.com](https://tracksingles.com/download/weights?secret=b02c29dc-606b-4f1e-8497-39c7e30c84b0) (lol)

## TPU stuff
(mostly notes to self from Nate)

When it's time to tear everything down, put this at the end of any `console.cloud.google.com/compute/` url: `&walkthrough_tutorial_id=cloud_tpu_quickstart`