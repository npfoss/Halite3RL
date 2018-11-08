#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction
from hlt.positionals import Position

# This library allows you to generate random numbers.
import random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging
import numpy as np
#import tensorflow as tf
#from tensorflow import keras


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



with open("weights") as f:
	model=pkl.load(f)


""" <<<Game Loop>>> """

while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    rounds_left = constants.MAX_TURNS - game.turn_number
    halite = np.array([[[game_map[Position(x,y)].halite_amount] for x in range(board_length)] for y in range(board_length)])
    friendly_ships_halite = np.array([[game_map[Position(x,y)].structure_type for x in range(board_length)] for y in range(board_length)])
    #np.set_printoptions(threshold='nan')
    logging.info(str(np.nonzero(friendly_ships_halite)))




    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []

    for ship in me.get_ships():

        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.
        if model.predict_classes([[halite]])[0] < 4 and (game_map[ship.position].halite_amount < constants.MAX_HALITE / 10 or ship.is_full):
            actions, mus, states = self.model._step(self.obs, S=self.states, M=self.dones)
            command_queue.append(
                ship.move(
                    [ Direction.North, Direction.South, Direction.East, Direction.West][actions[0]]))
                    # random.choice([ Direction.North, Direction.South, Direction.East, Direction.West ])))
        else:
            command_queue.append(ship.stay_still())

    # If the game is in the first 200 turns and you have enough halite, spawn a ship.
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 200 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

