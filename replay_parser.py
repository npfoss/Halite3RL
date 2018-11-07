import json
import zstd
import hlt
import numpy as np


def localize_matrix(m, new_length, old_center_x, old_center_y):
    # at most can double 
    old_length = m.shape[0]
    tall_m = np.concatenate((m, m,m), 0)   
    big_m = np.concatenate((tall_m, tall_m, tall_m), 1) 

    short_edge = int(np.floor((new_length - 1)/2))
    long_edge = int(np.ceil((new_length - 1)/2))

    new_center_x = old_center_x + old_length
    new_center_y = old_center_y + old_length 

    centered_m = big_m[new_center_y - short_edge: new_center_y + long_edge + 1, new_center_x - short_edge: new_center_x + long_edge + 1]
    return centered_m 

def load_replay(file_name, player_id):
	player_id = str(player_id)
    parsed_frames = []
    with open(file_name, 'rb') as f:
        data = json.loads(zstd.loads(f.read()))

    # ***** Unchanging constants *****
    board_length = data['production_map']['width']
    max_turns = data['GAME_CONSTANTS']['MAX_TURNS']
    actual_turns = len(data['full_frames'])

    # ***** Load initial state ***** 
    halite = np.zeros((board_length, board_length), dtype = np.int32)
    friendly_dropoffs = np.zeros((board_length, board_length), dtype = np.int32)
    enemy_dropoffs = np.zeros((board_length, board_length), dtype = np.int32)

    for y in range(board_length): #halite 
        for x in range(board_length):
            halite[y][x] = data['production_map']['grid'][y][x]['energy']

    for player_info in data['players']: # dropoffs 
        player = str(player_info['player_id'])
        x = player_info['factory_location']['x']
        y = player_info['factory_location']['y']    
        if (player == player_id): # friendly
            friendly_dropoffs[y][x] = 1
        else: #enemy
            enemy_dropoffs[y][x] = 1


    # ***** Update for each frame ***** 
    for t in range(actual_turns):
        frame = data['full_frames'][t]

        # Generate matrices for this turn 
        old_halite = np.copy(halite)
        friendly_ships = np.zeros((board_length, board_length), dtype = np.int32)
        friendly_ships_halite = np.zeros((board_length, board_length), dtype = np.int32)
        old_friendly_dropoffs = np.copy(friendly_dropoffs)
        enemy_ships = np.zeros((board_length, board_length), dtype = np.int32)
        enemy_ships_halite = np.zeros((board_length, board_length), dtype = np.int32)
        old_enemy_dropoffs = np.copy(enemy_dropoffs)

        rounds_left = max_turns - t
        player_energy = frame['energy'][player_id]

        for changed_cell in frame['cells']: # update halite 
            x = changed_cell['x']
            y = changed_cell['y']
            halite[y][x] = changed_cell['production']

        for player in frame['entities']: # update ships 
            player_entities = frame['entities'][player]
            for entity_id in player_entities:
                entity = player_entities[entity_id]
                x = entity['x']
                y = entity['y']
                energy = entity['energy']
                if (player == player_id): # friendly 
                    friendly_ships[y][x] = 1
                    friendly_ships_halite[y][x] = energy
                else: # enemy 
                    enemy_ships[y][x] = 1
                    enemy_ships_halite[y][x] = energy 

        for event in frame['events']: # update dropoffs 
            if event['type'] == 'construct':
                x = event['location']['x']
                y = event['location']['y']
                player = str(event['owner_id'])
                if (player == player_id): # friendly 
                    friendly_dropoffs[y][x] = 1
                else: # enemy 
                    enemy_dropoffs[y][x] = 1

        halite_left = old_halite.sum()  # This is different than the halite available on the
                                        # online thing.  Mine doesn't count onboard ship halite. 

        turn_results = {"halite": old_halite, "friendly_ships": friendly_ships, "friendly_ships_halite":friendly_ships_halite,
        				"friendly_dropoffs": old_friendly_dropoffs, "enemy_ships": enemy_ships, "enemy_ships_halite": enemy_ships_halite,
        				"enemy_dropoffs": old_enemy_dropoffs, "total_halite": sum(old_halite), "player_energy": player_energy, 
        				"rounds_left": rounds_left, "board_length": board_length}                           

        parsed_frames.append(turn_results)

    return parsed_frames
