import sys
import math
import random
import time
import copy
# To debug: print("Debug messages...", file=sys.stderr, flush=True)

"""

Passer le state en dict[tuple] plutôt qu'en list de list ?

Timeout append
Verif is_win avec les diagonales de 2

"""

Ns = {}     # Number of time a state has been visited
Nsa = {}    # Number of time a state / action pair has been visited
Pmcts = {}  # Number of points after taking a state / action pair
Qmcts = {}  # Quality of a state / action pair

Sa = {}     # Actions save -> key: (stateT, last_move), value: moves

# Hyperparameters
c = math.sqrt(2)

# Game state
game = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]
mini_game = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]

begin_time = 0

last_moves = None

time_save = []

# ----- FUNTIONS -----

# --- DEBUG ---
def print_board(board, mini_board):

    print(f"\nmini_board ->\n", file=sys.stderr, flush=True, end='')
    for y in range(3):
        for x in range(3):
            print(f"{mini_board[y][x]}", file=sys.stderr, flush=True, end='')
            if x != 2:
                print(f"|", file=sys.stderr, flush=True, end='')
        if y != 2:
            print(f"\n------\n", file=sys.stderr, flush=True, end='')

    print(f"\nBoard ->\n", file=sys.stderr, flush=True, end='')
    for y in range(9):
        for x in range(9):
            print(f"{board[y][x]}", file=sys.stderr, flush=True, end='')
            if x % 3 == 2 and x != 8:
                print(f"|", file=sys.stderr, flush=True, end='')
        print(f"\n", file=sys.stderr, flush=True, end='')
        if y % 3 == 2 and y != 8:
            print(f"------------\n", file=sys.stderr, flush=True, end='')


# --- GAME FUNCTIONS ---
def get_next_grid(last_move): #Valid
    """
    char    last_grid_x = (char)(last_play_x / 3) * 3;  //Select new grid based on opponent last play
    char    last_grid_y = (char)(last_play_y / 3) * 3;
    char    new_grid_x = last_play_x - last_grid_x;
    char    new_grid_y = last_play_y - last_grid_y;
    """
    # Select last grid coord
    last_grid = (int(last_move[0] / 3) * 3, int(last_move[1] / 3) * 3)
    
    # print(f"Last move {last_move} / last grid coords {last_grid}", file=sys.stderr, flush=True)

    # Vectors sub -> next coord grid
    return ((last_move[0] - last_grid[0]) * 3, (last_move[1] - last_grid[1]) * 3)

def get_moves_id(stateT, last_move, mcts=False): #Valid

    next_grid = None
    if mcts:
        print(f"get moves ids: {next_grid}", file=sys.stderr, flush=True)
        print_board(stateT, mini_state)
    
    if last_move[0] != -1:
        # Tuple (y, x)
        next_grid = get_next_grid(last_move)

        # Check if case is not finish
        if mini_state[int(next_grid[0] / 3)][int(next_grid[1] / 3)] == ' ':

            # if mcts:
            #     print(f"Next grid: {next_grid}", file=sys.stderr, flush=True)

            # Get coord of 1 grid
            ys = [next_grid[0], next_grid[0] + 1, next_grid[0] + 2]
            xs = [next_grid[1], next_grid[1] + 1, next_grid[1] + 2]

            # if all([all([state[y][x] != ' ' for x in xs]) for y in ys]):
            #     # Finish -> Tranform 1 grid in 9 grid
            #     ys = range(9)
            #     xs = range(9)
        else:
            ys = range(9)
            xs = range(9)
    else:
        ys = range(9)
        xs = range(9)
    
    # print(f"Empty moves ? : {ys} / {xs}", file=sys.stderr, flush=True)

    # Create ids of all moves
    moves = []
    # If at least one ' ' case exist ?
    # if any([any(True if state[y][x] == ' ' else False for x in xs) for y in ys]):

    # Append move if case != ' '
    # Need to fill all case witch are already won with his sign
    # [[moves.append((stateT, last_move, (y, x))) for x in xs if state[y][x] == ' '] for y in ys]
    [[moves.append((stateT, (y, x))) for x in xs if state[y][x] == ' '] for y in ys]

    # print(f"Possible moves ->", file=sys.stderr, flush=True)
    # for stateT, move in moves:
    #     print(f"{move} / ", file=sys.stderr, flush=True)
    return moves, next_grid

def fetch_moves_id(Said, mcts=False):
    """
    next_grid = None

    # print(f"MCTS {depth} / last_move {last_move}", file=sys.stderr, flush=True)
    # Save actions linked to stateT or fetch them
    if Said not in Sa.keys():

        # moves, next_grid = get_moves_id(*Said, mcts)
        moves, next_grid = get_moves_id(Said[0], Said[1], mcts)
        Sa[Said] = moves

    return Sa[Said], next_grid
    """
    return get_moves_id(Said[0], Said[1], mcts)

def apply_move(board, mini_board, move, sign, state_update=False): # To test
    
    # print_board(board, mini_board)
    try:
        y = move[0]
        x = move[1]
        board[y][x] = sign
    except:
        print(f"Apply move {move} sign {sign}", file=sys.stderr, flush=True)
        exit(1)

    # Get coord of this grid in mini_board
    mini_grid_y = int(y / 3)
    mini_grid_x = int(x / 3)

    # Get coord of this grid in board
    grid_y = mini_grid_y * 3
    grid_x = mini_grid_x * 3

    winner = is_grid_win(board, mini_board, grid_y, grid_x)
    if winner:
        # print_board(board, mini_board)
        mini_board[mini_grid_y][mini_grid_x] = sign

        # Fill 3x3
        # Need to fill all case witch are already won with his sign for get_moves_id selection
        for tmpy in [grid_y, grid_y + 1, grid_y + 2]:
            for tmpx in [grid_x, grid_x + 1, grid_x + 2]:
                board[tmpy][tmpx] = sign
    
    if state_update:
        reset_state()

def is_grid_win(board, mini_board, y1, x1): # To test

    # if coord_in_grid_y == 0:
    if board[y1][x1] != ' ':
        # Horizontale 1
        if board[y1][x1] == board[y1][x1 + 1] and board[y1][x1 + 1] == board[y1][x1 + 2]:
            return board[y1][x1]

        # Diagonale 00 to 22
        # if coord_in_grid_x == 0 and board[y1][x1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1 + 2][x1 + 2]:
        if board[y1][x1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1 + 2][x1 + 2]:
            return board[y1][x1]

    # elif coord_in_grid_y == 1:
    if board[y1 + 1][x1] != ' ':
        # Horizontale 2
        if board[y1 + 1][x1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1 + 1][x1 + 2]:
            return board[y1 + 1][x1]

    # else:
    if board[y1 + 2][x1] != ' ':
        # Horizontale 3
        if board[y1 + 2][x1] == board[y1 + 2][x1 + 1] and board[y1 + 2][x1 + 1] == board[y1 + 2][x1 + 2]:
            return board[y1 + 2][x1]

        # Diagonale 20 to 02
        # if coord_in_grid_x == 0 and board[y1 + 2][x1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1][x1 + 2]:
        if board[y1 + 2][x1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1][x1 + 2]:
            return board[y1][x1]

    # if coord_in_grid_x == 0:
    if board[y1][x1] != ' ':
        # Verticale 1
        if board[y1][x1] == board[y1 + 1][x1] and board[y1 + 1][x1] == board[y1 + 2][x1]:
            return board[y1][x1]

    # elif coord_in_grid_x == 1:
    if board[y1][x1 + 1] != ' ':
        # Verticale 2
        if board[y1][x1 + 1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1 + 2][x1 + 1]:
            return board[y1][x1 + 1]

    # else:
    if board[y1][x1 + 2] != ' ':
        # Verticale 3
        if board[y1][x1 + 2] == board[y1 + 1][x1 + 2] and board[y1 + 1][x1 + 2] == board[y1 + 2][x1 + 2]:
            return board[y1][x1 + 2]

    return 0


def is_win():

    if mini_state[0][0] != ' ':
        # Horizontale 1
        if mini_state[0][0] == mini_state[0][1] and mini_state[0][1] == mini_state[0][2]:
            return mini_state[0][0]

        # Diagonale 00 to 22
        if mini_state[0][0] == mini_state[1][1] and mini_state[1][1] == mini_state[2][2]:
            return mini_state[0][0]

    if mini_state[1][0] != ' ':
        # Horizontale 2
        if mini_state[1][0] == mini_state[1][1] and mini_state[1][1] == mini_state[1][2]:
            return mini_state[1][0]

    if mini_state[2][0] != ' ':
        # Horizontale 3
        if mini_state[2][0] == mini_state[2][1] and mini_state[2][1] == mini_state[2][2]:
            return mini_state[2][0]

        # Diagonale 20 to 02
        if mini_state[2][0] == mini_state[1][1] and mini_state[1][1] == mini_state[0][2]:
            return mini_state[2][0]

    if mini_state[0][0] != ' ':
        # Verticale 1
        if mini_state[0][0] == mini_state[1][0] and mini_state[1][0] == mini_state[2][0]:
            return mini_state[0][0]

    if mini_state[0][1] != ' ':
        # Verticale 2
        if mini_state[0][1] == mini_state[1][1] and mini_state[1][1] == mini_state[2][1]:
            return mini_state[0][1]

    if mini_state[0][2] != ' ':
        # Verticale 3
        if mini_state[0][2] == mini_state[1][2] and mini_state[1][2] == mini_state[2][2]:
            return mini_state[0][2]

    return 0

def tests_times(): # To test

    tmp_begin = time.time()
    
    ret = is_win()

    if len(time_save) < 1000:
        time_save.append(time.time() - tmp_begin)
    return ret


# --- SELECTION ---
def select_best_move_id(moves, next_grid, last_move, sign, depth): # Vali, last_moved

    # print(f"SELECTION nbr moves {len(moves)}", file=sys.stderr, flush=True)

    best_move = None
    best_UCB = -1000000
    random.shuffle(moves)
    for Nsaid in moves:

        # Nsid = (Nsaid[0], Nsaid[1])
        stateT = Nsaid[0]
        
        # Compute UCB value of this state/move pair
        try:
            # ns = Ns[Nsid]
            ns = Ns[stateT]
        except:
            print_board(stateT, mini_state)
            print(f"[CRASH STATET] last move {last_move}", file=sys.stderr, flush=True)
            print(f"[CRASH STATET] next grid {next_grid}", file=sys.stderr, flush=True)
            print(f"[CRASH STATET] All moves {[move[1] for move in moves]}", file=sys.stderr, flush=True)
            exit(1)
        try:
            nsa = Nsa[Nsaid]
            Q = Qmcts[Nsaid]
        except:
            print_board(stateT, mini_state)
            print_board(last_moves[0][0], mini_state)
            print(f"[CRASH STATET/ACTION] depth {depth}", file=sys.stderr, flush=True)
            print(f"[CRASH STATET/ACTION] last move {last_move}", file=sys.stderr, flush=True)
            print(f"[CRASH STATET/ACTION] next grid {next_grid}", file=sys.stderr, flush=True)
            print(f"[CRASH STATET/ACTION] last expensions moves {[move[1] for move in last_moves]}", file=sys.stderr, flush=True)
            print(f"[CRASH STATET/ACTION] All moves             {[move[1] for move in moves]}", file=sys.stderr, flush=True)
            exit(1)

        ucb = Q + c * math.sqrt(ns) / (1 + nsa)
        # ucb = Qmcts[id] + c * math.sqrt(Ns[stateT]) / (1 + Nsa[id])

        # Save the best
        if ucb > best_UCB:

            best_UCB = ucb
            best_move = Nsaid
    
    if best_UCB == 0:
        print(f"UCB: {best_UCB}", file=sys.stderr, flush=True)
        print(f"Moves: {[move[2] for move in moves]}", file=sys.stderr, flush=True)
        # print(f"UCB: {debug_ucb}", file=sys.stderr, flush=True)
        print(f"last_move: {last_move} / next_grid: {next_grid} / sign: {sign} / depth: {depth}", file=sys.stderr, flush=True)


    return best_move


# --- EXPANSION ---
# def expansion(Nsid, moves): # Valid
def expansion(stateT, moves): # Valid

    # print(f"EXPANSION", file=sys.stderr, flush=True)
    # print(f"EXPANSION state {stateT}", file=sys.stderr, flush=True)
    # Ns[Nsid] = 1
    Ns[stateT] = 1

    # for Nsaid in moves:
    #     Nsa[Nsaid] = 0
    #     Pmcts[Nsaid] = 0
    #     Qmcts[Nsaid] = 0 # Useless ??
    for y in range(9):
        for x in range(9):
            if stateT[y][x] == ' ':
                Nsaid = (stateT, (y, x))
                Nsa[Nsaid] = 0
                Pmcts[Nsaid] = 0
                Qmcts[Nsaid] = 0 # Useless ??


# --- SIMULATION / CNN ---
def simulation(last_move, sign): # To test

    # print(f"SIMULATION ->", file=sys.stderr, flush=True)
    stateT = tuple([tuple(row) for row in state])

    # Get possible moves
    moves, next_grid = fetch_moves_id((stateT, last_move))

    # Draw case
    if not moves:
        # print_board(state, mini_state)
        # print(f"SIMULATION END DRAW", file=sys.stderr, flush=True)
        return 0

    # Select one randomly
    move = random.choice(moves)[1]
    
    # Apply move
    apply_move(state, mini_state, move, sign)

    winner = is_win()
    if winner:
        # print_board(state, mini_state)
        # print(f"SIMULATION END winner = '{winner}'", file=sys.stderr, flush=True)
        return 1
    else:
        return -simulation(move, ('O' if sign == 'X' else 'X'))


# --- Monte Carlo Tree Search (Rave optimisation) ---

def MCTS(last_move, sign='X', depth=0):
    """
        policy, v = nn.predict(sArray.reshape(1,9,9))       # CNN prediction
        v = v[0][0]                                        
        valids = np.zeros(81)                               
        np.put(valids, possibleA, 1)                          # Matrix with possibilities as 1
        policy = policy.reshape(81) * valids                # Remove impossible move
        policy = policy / np.sum(policy)                    # Normalize ?
        P[sTuple] = policy                                  # Save prediction
    """
    #if depth > 4:
    #    print(f"MCTS {depth} / last_move {last_move}", file=sys.stderr, flush=True)
    # print(f"MCTS {depth} / last_move {last_move}", file=sys.stderr, flush=True)

    # print(f"state type {type(state)}", file=sys.stderr, flush=True)
    stateT = tuple([tuple(row) for row in state])
    # Nsid = (stateT, last_move)

    # moves, next_grid = fetch_moves_id(Nsid, mcts=False)
    moves, next_grid = fetch_moves_id((stateT, last_move), mcts=False)

    # print(f"Nbr moves {len(moves)}", file=sys.stderr, flush=True)

    if moves:
        # print(f"GO DEEPER", file=sys.stderr, flush=True)
        # [print(f"Ns: {key[0]}", file=sys.stderr, flush=True) for key in Ns.keys()]


        # Node already exist ?
        # if Nsid in Ns.keys():
        if stateT in Ns.keys():

            # print(f"if True", file=sys.stderr, flush=True)
            # - SELECTION
            best_move_id = select_best_move_id(moves, next_grid, last_move, sign, depth)
            move = best_move_id[1]

            apply_move(state, mini_state, move, sign)

            winner = is_win()
            if winner:
                points = 1 if winner == sign else -1
            else:
                points = MCTS(move, ('X' if sign == 'O' else 'O'), depth + 1)

            # print(f"BACKPROPAGATION depth {depth} / sign {sign} / points {points}", file=sys.stderr, flush=True)
            # - BACKPROPAGATION
            # Ns[Nsid] += 1
            Ns[stateT] += 1
            Nsa[best_move_id] += 1
            Pmcts[best_move_id] += points
            Qmcts[best_move_id] = Pmcts[best_move_id] / Nsa[best_move_id]
            return -points

        # Leaf node ?
        else:
            # print(f"if False", file=sys.stderr, flush=True)

            # - EXPENSION
            # expansion(Nsid, moves)
            expansion(stateT, moves)

            global last_moves
            last_moves = moves

            # print(f"SIMULATION ->", file=sys.stderr, flush=True)
            # print_board(state, mini_state)

            # - SIMULATION / CNN
            points = simulation(last_move, sign)

            # Why not save simulation result in Pmcts ? On based on child simulation ?
            return -points

    else:
        # No move left -> Draw
        # print(f"DRAW depth {depth}", file=sys.stderr, flush=True)
        return 0


# ----- MAIN FUNCTIONS -----
def reset_state():

    # print_board(state, mini_state)
    global state
    # state = [[sign for sign in row] for row in game]
    state = copy.deepcopy(game)
    global mini_state
    # mini_state = [[sign for sign in row] for row in mini_game]
    # print_board(state, mini_state)
    mini_state = copy.deepcopy(mini_game)

def console_tests():

    print_board(game, mini_game) # For console tests
    
    winner = is_win()
    if winner:
        print(f"WINNER IS {winner}")
        exit()

def parsing(sign='O'):

    # last_move = tuple([int(i) for i in input().split()])
    print(f"Your move -> ", file=sys.stderr, flush=True, end='\0')

    y = int(input())
    x = int(input())
    last_move = (y, x)

    if last_move[0] != -1:
        apply_move(game, mini_game, last_move, sign)
        print(f"Apply console move -> {last_move}", file=sys.stderr, flush=True)

    console_tests()
    """
    valid_action_count = int(input())
    valid_actions = []
    for i in range(valid_action_count):
        valid_actions.append(tuple([int(j) for j in input().split()]))
    """
    global begin_time
    begin_time = time.time()
    # print(f"begin_time -> {begin_time}", file=sys.stderr, flush=True)

    return last_move

def print_best_move(last_move, sign='X'):

    best_move = None
    best_value = -1000000

    gameT = tuple([tuple(row) for row in game])
    print_board(gameT, mini_game)
    reset_state()
    # for y in range(9):
    #     for x in range(9):
    #         Nsaid = (gameT, (y, x))

    #         # [print(f"Ns: {key[0][0]} / {key[1]} / {key[2]}", file=sys.stderr, flush=True) for key in Pmcts.keys()]
    #         # print(f"Ns: {[key[2] for key in Qmcts.keys()]}", file=sys.stderr, flush=True)

    #         # if Nsaid in Qmcts.keys() and Qmcts[id] > best_value:
    #         if Nsaid in Qmcts.keys():
                
    #             # print(f"Last move {Nsaid[1]} / Possible move -> {Nsaid[2]}", file=sys.stderr, flush=True)

    #             if Qmcts[Nsaid] > best_value:
    #                 best_move = Nsaid[1]
    #                 best_value = Qmcts[Nsaid]

    moves, next_grid = fetch_moves_id((gameT, last_move))
    print(f"PRINT BEST MOVE", file=sys.stderr, flush=True)
    print(f"Last move -> {last_move}", file=sys.stderr, flush=True)
    # print(f"Last move {last_move}", file=sys.stderr, flush=True)

    for Nsaid in moves:

        print(f"Possible move -> {Nsaid[1]}: {Qmcts[Nsaid]}", file=sys.stderr, flush=True)

        if Qmcts[Nsaid] > best_value:
            best_move = Nsaid[1]
            best_value = Qmcts[Nsaid]

    if best_move:
        print(f"{best_move[0]} {best_move[1]}")
        apply_move(game, mini_game, best_move, sign, state_update=True)
        print(f"Apply my move -> {best_move}", file=sys.stderr, flush=True)

        console_tests()
        return best_move
    else:
        print(f"ERROR NO BEST VALUE")
        exit(1)

# def tests():
#     pass


# ----- MAIN -----
# tests()
print_board(game, mini_game) # For console tests
# last_move = parsing()

last_move = (-1, -1)
sign = 'X'
turn_time = 0.990
while True:

    # print(f"begin / actual times - {begin_time} / {time.time()}", file=sys.stderr, flush=True)
    # print(f"current time -> {time.time() - begin_time}", file=sys.stderr, flush=True)
    i = 0
    # while time.time() - begin_time < turn_time:
    for k in range(1000):
        reset_state()

        # print(f"MONTE CARLO BEGIN {i}", file=sys.stderr, flush=True)
        MCTS(last_move, sign)
        # print(f"MONTE CARLO END {i}", file=sys.stderr, flush=True)
        i += 1

        # print(f"begin / actual times - {begin_time} / {time.time()}", file=sys.stderr, flush=True)
        # print(f"current time -> {time.time() - begin_time}", file=sys.stderr, flush=True)
        # print_board(game, mini_game)

    print(f"MCTS\tlast_move {last_move}\tnbr iter {i}", file=sys.stderr, flush=True)
    last_move = print_best_move(last_move, sign)
    # exit(0)
    sign = 'X' if sign == 'O' else 'O'
    # last_move = parsing(sign)
    # turn_time = 0.095
