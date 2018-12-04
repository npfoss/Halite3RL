#!/bin/sh

if [ $2 = 2 ]
then
    ./halite --replay-directory replays/ -vvv --no-timeout --results-as-json --width $1 --height $1 "python3 MyBot.py" "python3 MyBot.py"
else 
    ./halite --replay-directory replays/ -vvv --no-timeout --results-as-json --width $1 --height $1 "python3 MyBot.py" "python3 MyBot.py" "python3 MyBot.py" "python3 MyBot.py"
fi    