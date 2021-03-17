import sys
import tensorflow as tf

sys.path.append('../CNN_reinforcement')
sys.path.append('../random_simulation_rave')

import mcts_rave_CNN as mctscnn
import mcts_rave as mctsrd
import mcts_rave as mctsrd2
from CNN_connector import *

n_game = 1
mcts_iter = 500

cross_win = 0
model_name = "model_20_1000.mcts_rave"
model = RLModel(name=model_name, model_path=model_name)

def apply_move_both(move, sign):
	mctscnn.apply_move(mctscnn.game, mctscnn.mini_game, move, sign)
	mctsrd.apply_move(mctsrd.game, mctsrd.mini_game, move, sign)


def play_game(turns):

	turns[0][0].init_mcts()
	turns[1][0].init_mcts()

	last_move = (4, 4)
	player = turns[1][0]
	sign = turns[1][1]
	n_iter = turns[1][2]

	# Optimization for turn 1
	apply_move_both(last_move, sign)
	print(f"'{sign}' apply {last_move}")

	i = 0
	for k in range(n_iter):
		player.reset_state()
		player.MCTS(last_move, sign=sign, model=model)
		i += 1
	print(f"MCTS iter {i}", file=sys.stderr, flush=True)

	player.print_board(player.game, player.mini_game)

	while True:

		for player, sign, n_iter in turns:

			i = 0
			# while time.time() - begin_time < turn_time:
			for k in range(n_iter):
				player.reset_state()

				# print(f"MONTE CARLO BEGIN {i}", file=sys.stderr, flush=True)
				player.MCTS(last_move, sign=sign, model=model)
				# print(f"MONTE CARLO END {i}", file=sys.stderr, flush=True)
				i += 1

			print(f"MCTS {sign}\tlast_move {last_move}\tnbr iter {i}", file=sys.stderr, flush=True)

			player.reset_state()
			last_move = player.print_best_move(last_move, sign)

			apply_move_both(last_move, sign)
			player.print_board(player.game, player.mini_game) # For console tests

			winner = player.is_win(player.mini_game)
			if winner or all([all([player.mini_game[tmpy][tmpx] != ' ' for tmpx in range(3)]) for tmpy in range(3)]):
				if winner:
					print(f"WINNER IS {sign}")
				else:
					print(f"- DRAW -")
				# print(f"Percentage escape get_moves_id() -> {100 * stat_1 / (stat_1 + stat_2)} %")
				# print(f"Percentage save get_moves_id() -> {100 * stat_3 / (stat_2)} %")
				# print(f"Values -> {stat_1}\t{stat_2}\t{stat_3}")
				return (1 if winner == 'X' else -1) if winner else 0


for k in range(n_game):
	# cross_win += play_game([(mctscnn, 'O', mcts_iter), (mctsrd, 'X', mcts_iter)])
	cross_win += play_game([(mctsrd, 'O', mcts_iter), (mctscnn, 'X', mcts_iter)])
	# cross_win += play_game([(mctscnn, 'O', 500), (mctsrd, 'X', 500)])
	# cross_win += play_game([(mctsrd, 'O'), (mctscnn, 'X')])

# for k in range(n_game):
# 	cross_win += play_game([(mctsrd, 'O'), (mctscnn, 'X')])


print(f"END {n_game} GAMES")
print(f"SCORE {cross_win} / {n_game} GAMES")
# print(f"MCTS CNN 'X' -> {100 * cnn_wins / n_game} % wins")
# print(f"MCTS RDM 'O' -> {100 * rd_wins / n_game} % wins")
# print(f"             -> {100 * (n_game - cnn_wins - rd_wins) / n_game} % draw")
