import json
import zstd
import hlt
import numpy as np
import tensorflow as tf

# from IPython import embed

def localize_matrix(m, new_length, old_center_y, old_center_x):
    # at most can double 
    old_length = m.shape[0]
    tall_m = np.concatenate((m, m,m), 0)   
    big_m = np.concatenate((tall_m, tall_m, tall_m), 1) 

    short_edge = int(np.floor((new_length - 1)/2))
    long_edge = int(np.ceil((new_length - 1)/2))

    new_center_x = old_center_x + old_length
    new_center_y = old_center_y + old_length 

    centered_m = big_m[new_center_y - short_edge: new_center_y + long_edge + 1,
                       new_center_x - short_edge: new_center_x + long_edge + 1]
    return centered_m 

def load_replay(file_name, player_id):
    # TODO: also read in mus
    player_id = str(player_id)
    parsed_frames = []
    with open(file_name, 'rb') as f:
        data = json.loads(zstd.loads(f.read()).decode("utf-8"))

    # ***** Unchanging constants *****
    board_size = data['production_map']['width']
    max_turns = data['GAME_CONSTANTS']['MAX_TURNS']
    actual_turns = len(data['full_frames'])

    # ***** Load initial state ***** 
    halite = np.zeros((board_size, board_size), dtype = np.int32)
    friendly_dropoffs = np.zeros((board_size, board_size), dtype = np.int32)
    enemy_dropoffs = np.zeros((board_size, board_size), dtype = np.int32)

    for y in range(board_size): #halite 
        for x in range(board_size):
            halite[y][x] = data['production_map']['grid'][y][x]['energy']

    for player_info in data['players']: # dropoffs 
        player = str(player_info['player_id'])
        x = player_info['factory_location']['x']
        y = player_info['factory_location']['y']    
        if (player == player_id): # friendly
            friendly_dropoffs[y][x] = 1
        else: #enemy
            enemy_dropoffs[y][x] = 1
    
    ship_info = {}


    # ***** Update for each frame ***** 
    for t in range(actual_turns):
        frame = data['full_frames'][t]

        # Generate matrices for this turn 
        old_halite = np.copy(halite)
        friendly_ships = np.zeros((board_size, board_size), dtype = np.int32)
        friendly_ships_halite = np.zeros((board_size, board_size), dtype = np.int32)
        old_friendly_dropoffs = np.copy(friendly_dropoffs)
        enemy_ships = np.zeros((board_size, board_size), dtype = np.int32)
        enemy_ships_halite = np.zeros((board_size, board_size), dtype = np.int32)
        old_enemy_dropoffs = np.copy(enemy_dropoffs)
        ship_info = {}
        # move info keyed by ship id
        # NOTE: moves are what happened LAST FRAME (so do t+1 for moves this frame)
        # 99% confident in this, seriously
        moves = ({str(d['id']): d for d in data['full_frames'][t+1]['moves'][player_id] if 'id' in d} if player_id in data['full_frames'][t+1]['moves'] else {})\
                    if t < actual_turns-1 else {}

        rounds_left = max_turns - t
        player_energy = frame['energy'][player_id] # energy at BEGINNING of frame
        energy_delta = 0 if t > actual_turns - 2 else data['full_frames'][t + 1]['energy'][player_id] - player_energy

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
                    ship_info[entity_id] = {'pos': {'x': x, 'y': y}}
                    ship_info[entity_id]['energy'] = energy
                    # note: create supply depot move looks like this: {'id': 5, 'type': 'c'}
                    ship_info[entity_id]['action'] = 'o' if not entity_id in moves else \
                            moves[entity_id]['direction'] if 'direction' in moves[entity_id] else moves[entity_id]['type']
                    friendly_ships[y][x] = 1
                    friendly_ships_halite[y][x] = energy
                else: # enemy 
                    enemy_ships[y][x] = 1
                    enemy_ships_halite[y][x] = energy 

        if t > 0:
            # compute ship halite deltas for previous turn
            for idee in parsed_frames[t-1]['ship_info']:
                parsed_frames[t-1]['ship_info'][idee]['energy_delta'] = \
                                  ship_info[idee]['energy'] if idee in ship_info else 0 \
                                - parsed_frames[t-1]['ship_info'][idee]['energy']

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

        # remember that old mean "beginning of turn" aka current state during which actions were taken
        turn_results = {"halite_map"            : old_halite,
                        "friendly_ships"        : friendly_ships, # indicator variable
                        "friendly_ships_halite" : friendly_ships_halite,
                        "friendly_dropoffs"     : old_friendly_dropoffs, # indicator
                        "enemy_ships"           : enemy_ships,
                        "enemy_ships_halite"    : enemy_ships_halite,
                        "enemy_dropoffs"        : old_enemy_dropoffs,
                        "total_halite"          : old_halite.sum(), # total on the board
                        "player_energy"         : player_energy, # how much you have in the bank
                        "rounds_left"           : rounds_left,
                        "board_size"            : board_size,
                        "energy_delta"          : energy_delta,
                        # ** AUGMENTED WITH DELTA RETROACTIVELY
                        "ship_info"             : ship_info, # dictionary keyed by ship_id, contains (x,y) pos, energy, energy-delta, action and mu
                       }

        parsed_frames.append(turn_results)

    return parsed_frames


def replay_to_enc_obs_n_stuff(parsed_frames, gamma):
    """ converts output of load_replay to the format we store things in the buffer.

        right now, that's one giant string of observations+stuff, for each ship's
        journey through the game one after the other.
        (trace ends upon death, with reward on last frame taking into account the rest of the game)    

    used like this:
        enc_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks = replay_to_enc_obs_n_stuff(replay)
    """
    traces = {} # obs, actions, rewards, mus, dones, masks for each ship keyed by id
    for frame in parsed_frames:
        for ship_id, ship_info in frame["ship_info"].items():
            if ship_id not in traces:
                traces[ship_id]=[]
            traces[ship_id].append({
                "obs": gen_obs(frame, ship_info["pos"]),
                "actions": ship_info["action"],
                "rewards": gen_rewards(frame, ship_info),
                "mus": ship_info["mus"],
                "dones": False,
                "masks": False, # probably only matters for LSTMs, so... eh
            })
    for ship in traces.values():
        ship[-1]["dones"] = True
    return_order = ["obs", "actions", "rewards", "mus", "dones", "masks"]
    return [sum(([frame[key] for frame in ship] for ship in traces.values()), []) for key in return_order]

def gen_rewards(state, ship_info):
    # magic numbers
    ship_pickup_multiplier = 0.25

def gen_obs(state, ship_pos):
    """
    takes in the state (in replay-generated json format but also from the game)
    returns (64, 64, 7) tensor a la halite_env.py observation_space
    """
    obs = localize_matrix(state['halite_map'], 64, ship_pos['y'], ship_pos['x'])
    obs = np.repeat(obs, 7, axis=2)

    return obs
    # TODO: don't do that, delete it, it's wrong. modify below to do the right thing
    # TODO: also divide all halite amounts by 1000 because normalization?

    parsed_frames = state

    observations = []
    for frame_index, frame in enumerate(parsed_frames):
        frame_obs = []
        ship_locations = frame["friendly_ships"].nonzero() #array of x locs, array of y locs

        num_ships = len(ship_locations[0])
        
        for ship_index in range(num_ships):
            y = ship_locations[0][ship_index]
            x = ship_locations[1][ship_index]
            
            ship_obs = np.zeros((7,64,64))
            ship_action = 0
            ship_reward = 0

            ship_obs_index = 0
            keys_to_add = ["halite_map" , "friendly_ships" ,"friendly_ships_halite",
                           "friendly_dropoffs" ,"enemy_ships" ,"enemy_ships_halite",
                           "enemy_dropoffs" ]

            for key in keys_to_add:
                matrix = frame[key]
                ship_obs[ship_obs_index] = localize_matrix(matrix, 64,
                                                                 y, x)
                ship_obs_index += 1
                
            ship_obs = np.swapaxes(ship_obs, 0,1)
            ship_obs = np.swapaxes(ship_obs, 1, 2)
            frame_obs.append( (tf.convert_to_tensor(ship_obs), ship_action,
                               ship_reward) )

        observations.append(frame_obs)

    return observations

def enc_obs_to_obs(enc_obs):
    """ takes the encoded observations (output of replay_to_enc_obs) 

    Returns:
        list of observations in the form you would input them to the model
        list of actions corresponding to those observations
        list of rewards earned by taking those actions

    may be shaped funny to pretend there are multiple envs?

    TODO: also randomizes the orientation since there's symmetry and stuff.
        jk this is nontrivial, do it later. and don't forget about the actions
    """
    return enc_obs



if __name__ == "__main__":
    # for debugging
    # obs = load_observations("replays/ex_replay_2.hlt", "0")
    ...
