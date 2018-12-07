#!/bin/sh

if [ $2 = 2 ]
then
    ./halite --replay-directory replays/ -vvv --no-timeout --results-as-json --width $1 --height $1 "python3 MyBot.py --learning=true" "python3 MyBot.py --learning=true"
else 
    ./halite --replay-directory replays/ -vvv --no-timeout --results-as-json --width $1 --height $1 "python3 MyBot.py --learning=true" "python3 MyBot.py --learning=true" "python3 MyBot.py --learning=true" "python3 MyBot.py --learning=true"
fi    
