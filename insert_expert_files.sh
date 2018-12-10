#!/bin/sh

rm -rf expert_replays
mkdir expert_replays

python3 -m hlt_client.hlt_client replay user -i 2807 -l 50 -d expert_replays
# 2807 = teccles
# -l = num games
# -d = target directory


python3 hlt_to_phlt.py
./actor_upload.sh

rm -rf expert_replays

