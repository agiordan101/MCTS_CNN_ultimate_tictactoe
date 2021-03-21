
class Rave_policy():

	Ns = {}      # Number of time a state has been visited
	Nsa = {}     # Number of time a state / action pair has been visited
	Pmcts = {}   # Number of points after taking a state / action pair
	Qmcts = {}   # Quality of a state / action pair
	PCache = {}  # Prediction of CNN cached between every fit phase

	Sa = {}     # Moves save -> key: (stateT, last_move), value: moves

	def __init__(self):
		pass

	def update_state(self, state, action, heuristic):
		pass