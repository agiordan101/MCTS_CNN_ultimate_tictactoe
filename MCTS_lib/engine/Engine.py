from abc import ABCMeta, abstractmethod

class Engine(metaclass=ABCMeta):

	def __init__(self):
		pass

	@abstractmethod
	def is_win(self):
		pass

	@abstractmethod
	def apply_move(self, *args, **kwargs):
		pass

	def get_possible_move(self, *args, **kwargs):
		pass

	def heuristic(self, *args, **kwargs):
		pass
