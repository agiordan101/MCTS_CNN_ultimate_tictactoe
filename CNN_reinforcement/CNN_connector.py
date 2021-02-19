import numpy as np

all_yxs = []
for y in range(9):
	for x in range(9):
		all_yxs.append((y, x))

def fit(inputs, targets):
	print("Fit ...")


def convert_game_into_nparray(gameT, sign):
	
	game_np = np.zeros((9, 9))

	for y, x in all_yxs:
		if gameT[y][x] != ' ':
			game_np[y, x] = 1 if gameT[y][x] == sign else -1

	return game_np
