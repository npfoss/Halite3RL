import json
import zstd
import hlt
import numpy as np
from baselines.acer.halite_env import HaliteEnv


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

def load_replay(file_name, player_id, mus_are_known=True):
    player_id = str(player_id)
    parsed_frames = []
    with open(file_name, 'rb') as f:
        data = json.loads(zstd.loads(f.read()).decode("utf-8"))

    if mus_are_known:
        game_mus = read_mus(player_id)
    else:
        game_mus = None#fake_mus(player_id, data)

    '''
    Notes on replay format (keys of data['full_frames']):
    entities: ships on the map at the BEGINNING of the frame
    moves: moves taken by those entities THIS turn
    events: events that happened THIS turn
    cells: changes to cells THIS turn
    deposited: total depositied for all of time as of END of this frame
    energy: player energy at END of turn
    '''

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
        if mus_are_known:
            frame_mus = ([] if t == 0 else game_mus[t - 1]) if t <= len(game_mus) else \
                {j:None for i in frame["entities"] for j in frame["entities"][i]} # Mus are all none if no moves were made


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
        moves = {str(d['id']): d for d in data['full_frames'][t]['moves'][player_id] if 'id' in d} \
                    if player_id in data['full_frames'][t]['moves'] else {}


        """ Note to Nate and Kent:

        don't give a fuck about masks

        buffer is going to be flat, don't forget to grab one extra obs at the end when sampling

        it's ok that the obs after a ship dies is unrelated, next obs only matters when done == False (probably)

        don't aggregate retrace at the end yet because we think it ignores cutoffs and that's hard ?

        everything seems shifted one timestep in the replay. ship as entity shows up on the same frame its move shows up in
            and we moved the moves so ...? figure out what to do about this Monday December 3rd 2018
        """



        rounds_left = max_turns - t
        player_energy = data['full_frames'][t-1]['energy'][player_id] if t > 0 else 5000 # energy at END of frame
        energy_delta = frame['energy'][player_id] - player_energy

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
                    ship_info[entity_id]['energy_delta'] = 0 # will be overridden once we know what actually happened
                    # note: create supply depot move looks like this: {'id': 5, 'type': 'c'}
                    ship_info[entity_id]['action'] = 'o' if not entity_id in moves else \
                            moves[entity_id]['direction'] if 'direction' in moves[entity_id] else moves[entity_id]['type']
                    friendly_ships[y][x] = 1
                    friendly_ships_halite[y][x] = energy
                    assert (not mus_are_known or t == 0 or t >= actual_turns-1 or frame_mus[entity_id] is not None) , "mus are none!! :("
                    ship_info[entity_id]["mus"] = np.array(frame_mus[entity_id], dtype=np.float32) if mus_are_known else np.array([np.nan])
                else: # enemy
                    enemy_ships[y][x] = 1
                    enemy_ships_halite[y][x] = energy

        if t > 0:
            # compute ship halite deltas for previous turn
            for idee in parsed_frames[t-1]['ship_info']:
                parsed_frames[t-1]['ship_info'][idee]['energy_delta'] = \
                                  (ship_info[idee]['energy'] if idee in ship_info else 0) \
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
        turn_results = {"halite_map"            : old_halite,                               # beginning of turn
                        "friendly_ships"        : friendly_ships, # indicator variable      # beginning of turn
                        "friendly_ships_halite" : friendly_ships_halite,                    # beginning of turn
                        "friendly_dropoffs"     : old_friendly_dropoffs, # indicator        # beginning of turn
                        "enemy_ships"           : enemy_ships,                              # beginning of turn
                        "enemy_ships_halite"    : enemy_ships_halite,                       # beginning of turn
                        "enemy_dropoffs"        : old_enemy_dropoffs,                       # beginning of turn
                        "total_halite"          : old_halite.sum(), # total on the board    # beginning of turn
                        "player_energy"         : player_energy, # how much you have in the bank    # beginning of turn
                        "rounds_left"           : rounds_left,                              # beginning of turn
                        "board_size"            : board_size,
                        "energy_delta"          : energy_delta,                             #ON THIS TURN **************
                        # ** AUGMENTED WITH DELTA RETROACTIVELY
                        "ship_info"             : ship_info, # dictionary keyed by ship_id, contains (x,y) pos, energy, energy-delta, action and mu
                                                            #                                           Beg.    Beg.    on turn        on-turn      on-turn
                       }

        parsed_frames.append(turn_results)

    parsed_frames = parsed_frames[1:-1]
    return parsed_frames


def replay_to_enc_obs_n_stuff(parsed_frames, env, gamma):
    """ converts output of load_replay to the format we store things in the buffer.

        right now, that's one giant string of observations+stuff, for each ship's
        journey through the game one after the other.
        (trace ends upon death, with reward on last frame taking into account the rest of the game)

    used like this:
        enc_obs, mb_actions, mb_rewards, mb_mus, mb_dones, mb_masks = replay_to_enc_obs_n_stuff(replay)
    """
    traces = {} # obs, actions, rewards, mus, dones, masks for each ship keyed by id
    for frame_index, frame in enumerate(parsed_frames):
        for ship_id, ship_info in frame["ship_info"].items():
            if ship_id not in traces:
                traces[ship_id]=[]
            traces[ship_id].append({
                "obs": gen_obs(frame, ship_info["pos"]),
                "actions": env.action_to_num[ship_info["action"]],
                "rewards": gen_rewards(frame, ship_info, \
                    ship_id in parsed_frames[frame_index + 1]["ship_info"] if frame_index + 1 < len(parsed_frames) else False),
                "mus": ship_info["mus"],
                "dones": False,
                # "masks": False, # probably only matters for LSTMs, so... eh.
                    # if you ever want to use masks, save them in the buffer too
            })

    for ship in traces.values():
        ship[-1]["dones"] = True
    return_order = ["obs", "actions", "rewards", "mus", "dones",]# "masks"]
    return [np.array(sum(([frame[key] for frame in ship] for ship in traces.values()), [])) for key in return_order] + [None] #masks

def read_mus(player_id):
    mus = []
    with open("bot-"+str(player_id)+".log") as f:
        for line in f:
            if "mu:" in line:
                mu_str = line.split("mu:")[-1]
                mus.append(json.loads(mu_str))

    return mus

def fake_mus(player_id, data):
    uniform_mus = np.ones(6)/6
    mus_len = len(data['full_frames']) - 2
    mus=[]
    for t in range(mus_len):
        if player_id in data['full_frames'][t+1]['entities']:
            entity_list =data['full_frames'][t+1]['entities'][player_id]
            mus.append({j:uniform_mus for i in entity_list for j in entity_list})
        else:
            mus.append([])
    return mus

def gen_rewards(state, ship_info, survives):
    # magic numbers
    ship_pickup_multiplier = 0.1
    # selfish right now
    if ship_info["energy_delta"] == -ship_info["energy"] and survives:
        # dropped off all halite, potential drop off
        initial_halite = state["halite_map"][ship_info["pos"]["y"]][ship_info["pos"]["x"]]
        move_cost = round(0.1 * initial_halite)
        dropped_off = ship_info["energy"] - move_cost
    elif (ship_info["action"] == 'c'):
        dropped_off = state["halite_map"][ship_info["pos"]["y"]][ship_info["pos"]["x"]] + ship_info['energy'] - 4000
    else:
        dropped_off = 0
    return ship_info["energy_delta"] * ship_pickup_multiplier + dropped_off

def gen_obs(state, ship_pos):
    """g
    takes in the state (in replay-generated json format but also from the game)
    returns (64, 64, 7) tensor a la halite_env.py observation_space
    """


    map_list = ['halite_map', 'friendly_ships', 'friendly_ships_halite', 'friendly_dropoffs',
                    'enemy_ships', 'enemy_ships_halite', 'enemy_dropoffs']
    map_tensor = np.zeros((7, 64, 64))
    for index, feature in enumerate(map_list):
        map_tensor[index] = localize_matrix(state[feature], 64, ship_pos['y'], ship_pos['x'])
    map_tensor = np.moveaxis(map_tensor, 0, 2)

    return map_tensor


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
    obs = load_replay("replays/replay-20181207-160318-0500-1544216591-32-32.hlt", "1")
    replay_to_enc_obs_n_stuff(obs, HaliteEnv(), 0.99)

