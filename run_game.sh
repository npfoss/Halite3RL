#!/bin/sh

./halite --replay-directory replays/ -vvv --width 32 --height 32 "python3 MyBot.py --learning=true" "python3 MyBot.py --learning=true"
