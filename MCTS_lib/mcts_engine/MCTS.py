import sys
import math
import random
import time
import copy
from timer import timer
import numpy as np

from CNN_connector import *

all_yxs = []
for y in range(9):
	for x in range(9):
		all_yxs.append((y, x))

miness_inf = -float("inf")

class MCTS():

	# # IN THE CHILD CLASS ENGINE
	# game = [[' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '], [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ']]
	# mini_game = [[' ', ' ', ' '], [' ', ' ', ' '], [' ', ' ', ' ']]

	# signs = []
	# states = []
	# qualities = []

	def __init__(self, game_engine, mcts_iter=600, sign=None):
		self.game_engine = game_engine

	def get_best_move_id(self):
		pass

	def selection(self):
		pass

	def mcts_compute(self, sign, last_move):
		pass

	def get_current_best_move(self):
		pass

	def search(self):
		# for 1000
			# mcts_compute()
		
		# return get_current_best_move()
		pass
