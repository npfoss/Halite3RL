# Halite3RL
A reinforcement learning bot for the Halite III programming competition https://halite.io/

## install

```bash
git clone https://github.com/npfoss/Halite3RL.git
cd Halite3RL
pip3 install tensorflow scipy zstd gym dill gym[atari] joblib
curl -O https://halite.io/assets/downloads/Halite3_Python3_Linux-AMD64.zip
unzip Halite3_Python3_Linux-AMD64.zip -d tmp_halite
mv tmp_halite/halite .
rm -rf tmp_halite/ Halite3_Python3_Linux-AMD64.zip
```

## running

to run the actors which generate data:
`python3 actor.py`

to run the learner (only one at a time please):
`python3 -m baselines.run --alg=acer --env=PongNoFrameskip-v4`
...it's fine.
