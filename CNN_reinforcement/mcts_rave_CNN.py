import sys
import math
import random
import time
import copy
from timer import timer
import numpy as np

from CNN_connector import *

# To debug: print("Debug messages...", file=sys.stderr, flush=True)

"""

Passer le state en dict[tuple] plutôt qu'en list de list ?

Timeout append
Verif is_win avec les diagonales de 2

"""

Ns = {}      # Number of time a state has been visited
Nsa = {}     # Number of time a state / action pair has been visited
Pmcts = {}   # Number of points after taking a state / action pair
Qmcts = {}   # Quality of a state / action pair
PCache = {}  # Prediction of CNN cached between every fit phase

Sa = {}     # Moves save -> key: (stateT, last_move), value: moves

# Hyperparameters
c = 4				#4 because policy => [0, 1] so sqrt 2 is too small
miness_inf = -float("inf")

# Game state
game = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]
mini_game = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]

signs = []
states = []
qualities = []

all_yxs = []
for y in range(9):
	for x in range(9):
		all_yxs.append((y, x))

begin_time = 0

stat_1 = 0
stat_2 = 0
stat_3 = 0

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
@timer
def get_next_grid(last_move):
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

@timer
def get_moves_id(stateT, last_move, mcts=False):

	yxs = all_yxs
	next_grid = None

	if last_move[0] != -1:

		y_next_grid, x_next_grid = get_next_grid(last_move)
		next_grid = (y_next_grid, x_next_grid)

		# Check if case is not finish
		if mini_state[int(y_next_grid / 3)][int(x_next_grid / 3)] == ' ':

			# Get coord of all case in next grid
			yxs = [(y_next_grid, x_next_grid), (y_next_grid, x_next_grid + 1), (y_next_grid, x_next_grid + 2),
					(y_next_grid + 1, x_next_grid), (y_next_grid + 1, x_next_grid + 1), (y_next_grid + 1, x_next_grid + 2),
					(y_next_grid + 2, x_next_grid), (y_next_grid + 2, x_next_grid + 1), (y_next_grid + 2, x_next_grid + 2)]

	# Append move if case != ' '
	# Need to fill all case witch are already won with his sign
	return [(stateT, (y, x)) for y, x in yxs if state[y][x] == ' '], next_grid

# Strong with CNN, almost useless with simulation
@timer
def fetch_moves_id(Said, mcts=False):

	# next_grid = None

	# if Said in Sa:

	#     global stat_1
	#     stat_1 += 1
	#     return Sa[Said], next_grid
	
	# else:
	#     global stat_2
	#     stat_2 += 1
	#     # moves, next_grid = get_moves_id(*Said, mcts)
	#     moves, next_grid = get_moves_id(Said[0], Said[1], mcts)
		
	#     if mcts:
	#         Sa[Said] = moves
	#         global stat_3
	#         stat_3 += 1

	#     return moves, next_grid

	return get_moves_id(Said[0], Said[1], mcts)

@timer
def fill_grid(board, mini_board, mini_grid_y, mini_grid_x, all_grid, sign):
	# print_board(board, mini_board)
	mini_board[mini_grid_y][mini_grid_x] = sign

	# Fill 3x3
	# Need to fill all case witch are already won with his sign for get_moves_id() selection
	for y, x in all_grid:
		board[y][x] = sign

@timer
def apply_move(board, mini_board, move, sign):
	
	# print_board(board, mini_board)
	try:
		y = move[0]
		x = move[1]
		board[y][x] = sign
	except:
		print(f"[ERROR] Apply move {move} sign {sign}", file=sys.stderr, flush=True)
		exit(1)

	# Get coord of this grid in mini_board
	mini_grid_y = y // 3
	mini_grid_x = x // 3

	# Get coord of this grid in board
	grid_y = mini_grid_y * 3
	grid_x = mini_grid_x * 3

	# Get all coords of this grid in board
	all_grid = [(grid_y, grid_x), (grid_y, grid_x + 1), (grid_y, grid_x + 2),
				(grid_y + 1, grid_x), (grid_y + 1, grid_x + 1), (grid_y + 1, grid_x + 2),
				(grid_y + 2, grid_x), (grid_y + 2, grid_x + 1), (grid_y + 2, grid_x + 2)]

	# Check winner
	winner = is_grid_win(board, mini_board, grid_y, grid_x)

	# Check draw
	if winner or is_grid_draw(board, all_grid):

		# Draw        
		if not winner:
			sign = '#'

		fill_grid(board, mini_board, mini_grid_y, mini_grid_x, all_grid, sign)

@timer
def is_grid_win(board, mini_board, y1, x1):

	if board[y1][x1] != ' ':
		# Horizontale 1
		if board[y1][x1] == board[y1][x1 + 1] and board[y1][x1 + 1] == board[y1][x1 + 2]:
			return board[y1][x1]

		# Diagonale 00 to 22
		if board[y1][x1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1 + 2][x1 + 2]:
			return board[y1][x1]

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
		if board[y1 + 2][x1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1][x1 + 2]:
			return board[y1][x1]

	if board[y1][x1] != ' ':
		# Verticale 1
		if board[y1][x1] == board[y1 + 1][x1] and board[y1 + 1][x1] == board[y1 + 2][x1]:
			return board[y1][x1]

	if board[y1][x1 + 1] != ' ':
		# Verticale 2
		if board[y1][x1 + 1] == board[y1 + 1][x1 + 1] and board[y1 + 1][x1 + 1] == board[y1 + 2][x1 + 1]:
			return board[y1][x1 + 1]

	if board[y1][x1 + 2] != ' ':
		# Verticale 3
		if board[y1][x1 + 2] == board[y1 + 1][x1 + 2] and board[y1 + 1][x1 + 2] == board[y1 + 2][x1 + 2]:
			return board[y1][x1 + 2]

	return 0

@timer
def is_grid_draw(board, all_grid):
	for y, x in all_grid:
		if board[y][x] == ' ':
			return 0
	return 1

@timer
def is_win(mini_board):

	if mini_board[0][0] != ' ':
		# Horizontale 1
		if mini_board[0][0] == mini_board[0][1] and mini_board[0][1] == mini_board[0][2]:
			return mini_board[0][0]

		# Diagonale 00 to 22
		if mini_board[0][0] == mini_board[1][1] and mini_board[1][1] == mini_board[2][2]:
			return mini_board[0][0]

	if mini_board[1][0] != ' ':
		# Horizontale 2
		if mini_board[1][0] == mini_board[1][1] and mini_board[1][1] == mini_board[1][2]:
			return mini_board[1][0]

	if mini_board[2][0] != ' ':
		# Horizontale 3
		if mini_board[2][0] == mini_board[2][1] and mini_board[2][1] == mini_board[2][2]:
			return mini_board[2][0]

		# Diagonale 20 to 02
		if mini_board[2][0] == mini_board[1][1] and mini_board[1][1] == mini_board[0][2]:
			return mini_board[2][0]

	if mini_board[0][0] != ' ':
		# Verticale 1
		if mini_board[0][0] == mini_board[1][0] and mini_board[1][0] == mini_board[2][0]:
			return mini_board[0][0]

	if mini_board[0][1] != ' ':
		# Verticale 2
		if mini_board[0][1] == mini_board[1][1] and mini_board[1][1] == mini_board[2][1]:
			return mini_board[0][1]

	if mini_board[0][2] != ' ':
		# Verticale 3
		if mini_board[0][2] == mini_board[1][2] and mini_board[1][2] == mini_board[2][2]:
			return mini_board[0][2]

	return 0


# --- SELECTION ---
@timer
def select_best_move_id(moves, policy, next_grid, last_move, sign, depth):

	# print(f"SELECTION nbr moves {len(moves)}", file=sys.stderr, flush=True)
	best_move = None
	best_UCB = miness_inf
	random.shuffle(moves)

	for Nsaid in moves:

		# Compute UCB value of this state/move pair
		ucb = Qmcts[Nsaid] + c * policy[Nsaid[1][0]][Nsaid[1][1]] * math.sqrt(Ns[Nsaid[0]]) / (1 + Nsa[Nsaid])

		# Save the best
		if ucb > best_UCB:

			best_UCB = ucb
			best_move = Nsaid

	return best_move


# --- EXPANSION ---
@timer
def expansion(stateT, moves):

	# print(f"EXPANSION", file=sys.stderr, flush=True)
	Ns[stateT] = 1

	for y, x in all_yxs:
		if stateT[y][x] == ' ':
			Nsaid = (stateT, (y, x))
			Nsa[Nsaid] = 0
			Pmcts[Nsaid] = 0
			Qmcts[Nsaid] = 0 # Useless ??


# --- SIMULATION / CNN ---
@timer
def simulation(last_move, sign):

	# print(f"SIMULATION {sign} ->", file=sys.stderr, flush=True)
	# print_board(state, mini_state)
	stateT = tuple([tuple(row) for row in state])

	# Get possible moves
	moves, next_grid = fetch_moves_id((stateT, last_move))

	# Draw case
	if not moves:
		# print(f"SIMULATION END DRAW", file=sys.stderr, flush=True)
		return 0

	# Select one randomly
	move = random.choice(moves)[1]
	
	# Apply move
	apply_move(state, mini_state, move, sign)

	return 1 if is_win(mini_state) else -simulation(move, ('O' if sign == 'X' else 'X'))


# --- Monte Carlo Tree Search (Rave optimisation) ---
@timer
def MCTS(last_move, sign, model, depth=0):

	# print("model", model)
	#if depth > 4:
	#    print(f"MCTS {depth} / last_move {last_move}", file=sys.stderr, flush=True)
	# print(f"MCTS {depth} / last_move {last_move}", file=sys.stderr, flush=True)

	# print(f"state type {type(state)}", file=sys.stderr, flush=True)
	stateT = tuple([tuple(row) for row in state])

	moves, next_grid = fetch_moves_id((stateT, last_move), mcts=True)
	# print(f"Nbr moves {len(moves)}", file=sys.stderr, flush=True)

	if moves:
		# print(f"GO DEEPER", file=sys.stderr, flush=True)

		if stateT not in PCache:
			mask = create_mask(moves)
			game_np = convert_game_into_nparray(stateT, sign)

			# print(f"mask {mask.shape}")
			# print(f"game_np {game_np.shape}")

			features = np.stack([game_np, mask], axis=-1)
			# print(f"features {features.shape}")

			policy, win = model.predict(features[np.newaxis, :, :, :])

			policy = policy.reshape(9, 9)
			policy *= mask
			win = win[0, 0]

			PCache[stateT] = policy, win

		policy, win = PCache[stateT]

		# if stateT not in PCache:
		# 	PCache[stateT] = model.predict(convert_game_into_nparray(stateT, sign)[np.newaxis, :, :, :])
		# policy, win = PCache[stateT]
		# policy = policy.reshape(9, 9)
		# policy *= create_mask(moves)
		# print(f"policy: {policy}")
		# win = win[0, 0]

		# Node already exist ?
		if stateT in Ns:

			# print(f"if True", file=sys.stderr, flush=True)
			# - SELECTION
			# best_move_id = select_best_move_id(moves, None, next_grid, last_move, sign, depth)
			best_move_id = select_best_move_id(moves, policy, next_grid, last_move, sign, depth)
			move = best_move_id[1]

			apply_move(state, mini_state, move, sign)

			points = 1 if is_win(mini_state) else MCTS(move, ('X' if sign == 'O' else 'O'), model, depth + 1)
			# print(f"BACKPROPAGATION depth {depth} / points {points}", file=sys.stderr, flush=True)

			# - BACKPROPAGATION
			Ns[stateT] += 1
			Nsa[best_move_id] += 1
			Pmcts[best_move_id] += points
			Qmcts[best_move_id] = Pmcts[best_move_id] / Nsa[best_move_id]
			return -points

		# Leaf node ?
		else:
			# print(f"if False", file=sys.stderr, flush=True)

			# - EXPENSION
			expansion(stateT, moves)

			# print(f"SIMULATION ->", file=sys.stderr, flush=True)

			# - SIMULATION / CNN
			# return -simulation(last_move, sign)
			return 1 if win < 0 else -1

	else:
		# No move left -> Draw
		return 0

@timer
def mcts_get_qualities(moves): # Tous les move possible pas ue ceux de ce tour là

	qualities = np.zeros((9, 9, 1))

	for move in moves:
		y, x = move[1]
		qualities[y, x, 0] = Nsa[move] / Ns[move[0]]

	qualities = qualities / np.sum(qualities)

	# print(f"Qualities {qualities}")
	# print(f"Feature -1: {states[-1]}")
	return qualities

@timer
def print_best_move(last_move, sign):

	best_move = None
	best_value = -1000000
	best_move_2 = None
	best_value_2 = -1000000

	gameT = tuple([tuple(row) for row in game])

	# print(f"fetch moves id to print best choice {last_move}\n{gameT}", file=sys.stderr, flush=True)
	reset_state()
	moves, next_grid = fetch_moves_id((gameT, last_move))

	print(f"Nbr moves {len(moves)}", file=sys.stderr, flush=True)
	for Nsaid in moves:

		# print(f"Possible move -> {Nsaid[1]}: {Qmcts[Nsaid]}\t= {Pmcts[Nsaid]}\t/ {Nsa[Nsaid]}\tPolicy: {PCache[gameT][0]}", file=sys.stderr, flush=True)
		print(f"Possible move -> {Nsaid[1]}: {Qmcts[Nsaid]}\t= {Pmcts[Nsaid]}\t/ {Nsa[Nsaid]}", file=sys.stderr, flush=True)

		if Qmcts[Nsaid] > best_value:
			best_move = Nsaid[1]
			best_value = Qmcts[Nsaid]

		if Nsa[Nsaid] > best_value_2:
			best_move_2 = Nsaid[1]
			best_value_2 = Nsa[Nsaid]

	if best_move != best_move_2:
		print(f"BEST MOVE ARE NOT EQUAL : Qmcts={best_move} / Nsa={best_move_2}")
		best_move = best_move_2

	if best_move:
		print(f"{best_move[0]} {best_move[1]}")
		print(f"'{sign}' apply {best_move}", file=sys.stderr, flush=True)

		signs.append(sign)
		states.append(np.stack([convert_game_into_nparray(gameT, sign), create_mask(moves)], axis=-1))
		qualities.append(mcts_get_qualities(moves).flatten())
		# print(type(qualities), type(qualities[-1]))
		# quit()
		# print(f"last npstate: {states[-1]}")

		apply_move(game, mini_game, best_move, sign)
		# reset_state()

		return best_move

	else:
		print(f"[ERROR MCTS RAVE CNN] NO BEST VALUE")
		exit(1)


# ----- MAIN FUNCTIONS -----
def reset_state():

	global state
	state = copy.deepcopy(game)
	global mini_state
	mini_state = copy.deepcopy(mini_game)

def parsing(sign='X'):

	# Codingame parsing
	# last_move = tuple([int(i) for i in input().split()])
	# print(f"Opponent move -> {last_move}", file=sys.stderr, flush=True)

	# Console parsing
	print(f"Your move -> ", file=sys.stderr, flush=True, end='\0')
	y = int(input())
	x = int(input())
	last_move = (y, x)

	if last_move[0] != -1:
		apply_move(game, mini_game, last_move, sign)
		reset_state()

		winner = is_win(mini_state)
		if winner or all([all([mini_game[tmpy][tmpx] != ' ' for tmpx in range(3)]) for tmpy in range(3)]):
			if winner:
				print(f"WINNER IS {winner}")
			else:
				print(f"- DRAW -")
			# print(f"Percentage escape get_moves_id() -> {100 * stat_1 / (stat_1 + stat_2)} %")
			# print(f"Percentage save get_moves_id() -> {100 * stat_3 / (stat_2)} %")
			# print(f"Values -> {stat_1}\t{stat_2}\t{stat_3}")
			exit(0)

	# Codingame parsing
	#Useless (Save valid action in Sa ?)
	# valid_action_count = int(input())
	# valid_actions = []
	# for i in range(valid_action_count):
	#     valid_actions.append(tuple([int(j) for j in input().split()]))

	global begin_time
	begin_time = time.time()
	# print(f"begin_time -> {begin_time}", file=sys.stderr, flush=True)

	return last_move

def init_mcts():

	print("-- INIT MCTS --")
	global Ns
	global Nsa
	global Pmcts
	global Qmcts
	global Sa
	global PCache
	Ns = {}
	Nsa = {}
	Pmcts = {}
	Qmcts = {}
	Sa = {}
	PCache = {}

	global signs
	global states
	global qualities
	signs = []
	states = []
	qualities = []

	global game
	global mini_game
	game = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]
	mini_game = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]
	reset_state()

	print_board(game, mini_game)
