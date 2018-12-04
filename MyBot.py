#!/usr/bin/env python3
# Python 3.6

# NO PRINTING DURING IMPORTS DAMMIT
import os
import sys
f = open(os.devnull, 'w')
oldstdout = sys.stdout
sys.stdout = f
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# check if it's the real deal or just learning mode:
ITS_THE_REAL_DEAL = '--learning=true' in sys.argv

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


from baselines.acer.acer import Model, create_model
from baselines.common.tf_util import load_variables
import dill as pkl
import json

with open("params.pkl", "rb") as f:
    learn_params = pkl.load(f)
    env , policy , nenvs , ob_space , ac_space , nstack , model = create_model(**learn_params)
load_variables("actor.ckpt")

f.close()
sys.stdout = oldstdout

from replay_parser import localize_matrix, gen_obs

""" <<<Game Begin>>> """

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("MyPythonBot")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))
board_length = game.game_map.width


""" <<<Game Loop>>> """

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    rounds_left = constants.MAX_TURNS - game.turn_number
    halite = np.array([[game_map[Position(x,y)].halite_amount for x in range(board_length)] for y in range(board_length)])
    friendly_ships_halite = np.array([[game_map[Position(x,y)].structure_type for x in range(board_length)] for y in range(board_length)])

    map_list = ['halite_map', 'friendly_ships', 'friendly_ships_halite', 'friendly_dropoffs',
                    'enemy_ships', 'enemy_ships_halite', 'enemy_dropoffs']
    state = {
                'halite_map': halite,
                'friendly_ships': halite,
                'friendly_ships_halite': halite,
                'friendly_dropoffs': halite,
                'enemy_ships': halite,
                'enemy_ships_halite': halite,
                'enemy_dropoffs': halite,
            }

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    # halite_exp = np.expand_dims(halite, axis=1)

    frame_mus = {}
    for ship in me.get_ships():

        obs = gen_obs(state, {'x': ship.position.x, 'y': ship.position.y}) # (64,64,7)
        # print(halite.shape, obs.shape) # spoiler it's (32, 32, 1) (64, 64, 7)

        actions, mus, _ = model._step(obs)#, M=self.dones)
        # print('actions:', actions,\
        #         '\nmus:', mus,\
        #         '\nstates:', states)
        ## spoilers:
        ## actions: [1] 
        ## mus: [[0.16987674 0.16746981 0.15873784 0.16824721 0.168809   0.16685943]] 
        ## states: []
        # TODO: save mus (could just use log file!)
        frame_mus[ship.id] = [float(mu) for mu in mus[0]]
        action = actions[0]
        if action < 4:# and (game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full):
            command_queue.append(
                ship.move(
                    [Direction.North, Direction.South, Direction.East, Direction.West][action]))
        elif action == 4:
            command_queue.append(ship.stay_still())
        elif ship.halite_amount + game_map[ship.position].halite_amount + me.halite_amount >= 4000\
                and not game_map[ship.position].has_structure:
            # have to check because it crashes otherwise
            command_queue.append(ship.make_dropoff())
            me.halite_amount -= 4000 - (ship.halite_amount + game_map[ship.position].halite_amount)
    logging.info("mu:"+json.dumps(frame_mus))


    # **** SPAWN RULE STUFF **** 
    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())



    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

