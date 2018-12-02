#!/bin/sh

# pip install --target . gym
# # export PATH="/home/bot_compilation/.local/bin:$PATH"
# # python3.6 -m pip install --target . gym
# export PYTHONPATH="$PWD/:$PYTHONPATH"

# chmod -R 777 .
python3.6 -m pip install --target . tqdm
python3.6 -m pip install --target . zstd
