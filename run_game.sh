#!/bin/sh

./halite --replay-directory replays/ --no-timeout -vvv --width 32 --height 32 "python3 MyBot.py --learning=true" "python3 MyBot.py --learning=true"
