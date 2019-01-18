#!/usr/bin/env python3
# Python 3.6

exporation_proportion = 0.2

from benchmarking import benchmarker
from unittest.mock import Mock
benchmark = benchmarker(printer=lambda x: None, warner=lambda x: None)
benchmark.start()

# NO PRINTING DURING IMPORTS DAMMIT
import os
import sys
devnull = open(os.devnull, 'w')
oldstdout = sys.stdout
sys.stdout = devnull
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# check if it's the real deal or just learning mode:
ITS_THE_REAL_DEAL = '--learning=true' not in sys.argv
if ITS_THE_REAL_DEAL:
    exporation_proportion = 0

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction
from hlt.positionals import Position

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging
import numpy as np

from baselines.acer.acer import Model, create_model # this is what's taking 4 seconds
from baselines.common.tf_util import load_variables

import dill as pkl
import json
import random
if not ITS_THE_REAL_DEAL:
    benchmark.benchmark("imports")

with open("model_params.pkl", "rb") as f:
    learn_params = pkl.load(f)
    env , policy , nenvs , ob_space , ac_space , nstack , model = create_model(**learn_params)

if not ITS_THE_REAL_DEAL: benchmark.benchmark("create model")

load_variables("actor.ckpt")

if not ITS_THE_REAL_DEAL: benchmark.benchmark("load weights")

devnull.close()
sys.stdout = oldstdout

from replay_parser import gen_obs

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyPythonBot")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
if not ITS_THE_REAL_DEAL: logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))
board_length = game.game_map.width

if not ITS_THE_REAL_DEAL: benchmark.end("init done")
if not ITS_THE_REAL_DEAL: logging.info(str(benchmark))
if not ITS_THE_REAL_DEAL: benchmark = benchmarker()
if not ITS_THE_REAL_DEAL: benchmark.start()


""" <<<Game Loop>>> """

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    if not ITS_THE_REAL_DEAL: benchmark.benchmark("start turn")
    me = game.me
    game_map = game.game_map

    rounds_left = constants.MAX_TURNS - game.turn_number
    halite = np.array([[game_map[Position(x,y)].halite_amount for x in range(board_length)] for y in range(board_length)])
    friendly_ships = np.zeros((board_length, board_length))
    friendly_ships_halite = np.zeros((board_length, board_length))
    friendly_dropoffs = np.zeros((board_length, board_length))
    enemy_ships = np.zeros((board_length, board_length))
    enemy_ships_halite = np.zeros((board_length, board_length))
    enemy_dropoffs =  np.zeros((board_length, board_length))

    for player in game.players:
        player_ships = game.players[player].get_ships()
        player_dropoffs = game.players[player].get_dropoffs()

        for ship in player_ships:
            if player == me:
                friendly_ships[ship.position.y][ship.position.x] = 1
                friendly_ships_halite[ship.position.y][ship.position.x] = ship.halite_amount
            else:
                enemy_ships[ship.position.y][ship.position.x] = 1
                enemy_ships_halite[ship.position.y][ship.position.x] = ship.halite_amount

        for dropoff in player_dropoffs:
            if player == me:
                friendly_dropoffs[dropoff.position.y][dropoff.position.x] = 1
            else:
                enemy_dropoffs[dropoff.position.y][dropoff.position.x] = 1
        if game.players[player] == me: # shipyards don't count as dropoffs
            friendly_dropoffs[me.shipyard.position.y][me.shipyard.position.x] = 1
        else:
            enemy_dropoffs[game.players[player].shipyard.position.y][game.players[player].shipyard.position.x] = 1


    map_list = ['halite_map', 'friendly_ships', 'friendly_ships_halite', 'friendly_dropoffs',
                    'enemy_ships', 'enemy_ships_halite', 'enemy_dropoffs']


    state = {
                'halite_map': halite,
                'friendly_ships': friendly_ships,
                'friendly_ships_halite': friendly_ships_halite,
                'friendly_dropoffs': friendly_dropoffs,
                'enemy_ships': enemy_ships,
                'enemy_ships_halite': enemy_ships_halite,
                'enemy_dropoffs': enemy_dropoffs,
            }

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    # halite_exp = np.expand_dims(halite, axis=1)

    frame_mus = {}

    if not ITS_THE_REAL_DEAL: benchmark.benchmark("created env variables")
    returned_to_home = {}
    for ship in me.get_ships():

        obs = gen_obs(state, {'x': ship.position.x, 'y': ship.position.y}) # (64,64,7)
        # print(halite.shape, obs.shape) # spoiler it's (32, 32, 1) (64, 64, 7)

        if not ITS_THE_REAL_DEAL: benchmark.benchmark("generated observations")

        actions, mus, _ = model._step(obs)#, M=self.dones)

        if not ITS_THE_REAL_DEAL: benchmark.benchmark("ran model")
        frame_mus[ship.id] = [float(mu) for mu in mus[0]]
        if not ITS_THE_REAL_DEAL and random.random() < exporation_proportion:
            action = random.randrange(5) # NOTE: making this 4 excludes dropoffs
            logging.info('Moving randomly')
        else:
            action = actions[0]
        if action < 4:# and (game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full):
            command_queue.append(
                ship.move(
                    [Direction.North, Direction.South, Direction.East, Direction.West][action]))
        elif action == 4:
            command_queue.append(ship.stay_still())
        elif action == 5:
            # move towards home greedily
            closest = min(me.get_dropoffs(), key=lambda dropoff: abs(dropoff.position.x - ship.position.x) + abs(dropoff.position.y - ship.position.y))
            towards_base = []
            if abs(ship.position.x - closest.position.x) > len(game_map) / 2:
                # need to wrap around
                if closest.position.x < ship.position.x:
                    towards_base.append((1, 0, Direction.East))
                elif closest.position.x > ship.position.x:
                    towards_base.append((-1, 0, Direction.West))
            else:
                # don't need to wrap around
                if closest.position.x < ship.position.x:
                    towards_base.append((-1, 0, Direction.West))
                elif closest.position.x > ship.position.x:
                    towards_base.append((1, 0, Direction.East))
            if abs(ship.position.y - closest.position.y) > len(game_map[0]) / 2:
                # need to wrap around
                if closest.position.y < ship.position.y:
                    towards_base.append((0, 1, Direction.South))
                elif closest.position.y > ship.position.y:
                    towards_base.append((0, -1, Direction.North))
            else:
                # don't need to wrap around
                if closest.position.y < ship.position.y:
                    towards_base.append((0, -1, Direction.North))
                elif closest.position.y > ship.position.y:
                    towards_base.append((0, 1, Direction.South))
            home_action = min(towards_base, key=lambda delta: game_map[(ship.position.x + delta[0], ship.position.y + delta[1])])[2]
            command_queue.append(ship.move(home_action))
        elif action == 6 and ship.halite_amount + game_map[ship.position].halite_amount + me.halite_amount >= 4000\
                and not game_map[ship.position].has_structure:
            # have to check because it crashes otherwise
            command_queue.append(ship.make_dropoff())
            me.halite_amount -= 4000 - (ship.halite_amount + game_map[ship.position].halite_amount)

        returned_to_home[ship.id] = bool(action == 5)

        # documentation ?
        if not ITS_THE_REAL_DEAL: benchmark.benchmark("ship {} turn stats".format(ship.id))


    # **** SPAWN RULE STUFF ****
    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())


    if not ITS_THE_REAL_DEAL:
        logging.info("mu:"+json.dumps(frame_mus))
        logging.info("rth:"+json.dumps(returned_to_home))
        benchmark.benchmark("end turn")
    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
