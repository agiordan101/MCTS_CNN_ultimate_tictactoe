import numpy as np
import tensorflow as tf

all_yxs = []
for y in range(9):
	for x in range(9):
		all_yxs.append((y, x))

def convert_game_into_nparray(gameT, sign):
	
	game_np = np.zeros((9, 9))

	for y, x in all_yxs:
		if gameT[y][x] != ' ':
			game_np[y, x] = 1 if gameT[y][x] == sign else -1

	# print(f"gameT: {gameT}")
	# print(f"sign: {sign}")
	# print(f"game_np: {game_np}")
	# print(game_np.shape)
	# quit()
	state = game_np[:, :, np.newaxis]
	# print(f"state: {state}")
	return state


class RLModel:

	def __init__(self, name=None):
		self.name = name or 'model'
		# if name:
		# 	self.model = tf.keras.models.load_model(name)
		# else:
		self.model = self.get_model()

	def get_model(self):
		i = tf.keras.layers.Input(shape=(9, 9, 1))
		conv = tf.keras.layers.Conv2D(
			filters=16,
			strides=(3, 3),
			kernel_size=(3, 3),
			activation='linear')(i)
		flatten = tf.keras.layers.Flatten()(conv)
		dense1 = tf.keras.layers.Dense(512, activation='linear')(flatten)
		dense2 = tf.keras.layers.Dense(256, activation='linear')(dense1)

		policy = tf.keras.layers.Dense(81, activation='tanh', name='p')(dense2)
		value = tf.keras.layers.Dense(1, activation='tanh', name='v')(dense2)

		# conv = la
		model = tf.keras.Model(inputs=i, outputs=[policy, value])
		model.summary()
		model.compile(
			loss='MSE',
			optimizer='Adam',
			metrics=['accuracy']
		)
		model.summary()
		return model

	def fit(self, features, targets, epochs=100):
		return self.model.fit(features, targets, epochs=epochs)
	
	def predict(self, features):
		return self.model.predict(features)

def parse(dataset_path):

	with open(dataset_path, 'r') as f:
		# dataset_str = f.read()
		# dataset_lst = dataset_str.split('\n')
		states = []
		qualities = []
		values = []
		for row in f.read().split('\n'):
			srow = row.split(',')
			# print(srow)
			if len(srow) == 163:
				states.append(np.array(srow[:81]).astype(float).reshape(9, 9, 1))
				qualities.append(np.array(srow[81:162]).astype(float))
				values.append(np.array(srow[162:]).astype(float))
			# if np.array(states[-1]).shape != (81,):
			# 	print(srow)
			# 	print(np.array(states[-1]).shape)
			# if np.array(qualities[-1]).shape != (81,):
			# 	print(np.array(qualities[-1]).shape)
			# if np.array(values[-1]).shape != (1,):			
			# 	print(np.array(values[-1]).shape)			
		# print(np.array(states).shape)
		# print(np.array(qualities).shape)
		# print(np.array(values).shape)
		return np.array(states), [np.array(qualities), np.array(values)]
		# dataset_lst = [row.split(',') for row in dataset_lst]

if __name__ == '__main__':

	model = RLModel()

	features, targets = parse('dataset.mcts')
	model.fit(features, targets)
	print(model.predict(features[:1])[0])
	print(model.predict(features[:1])[1])
	print(targets[0][:1])
	print(targets[1][:1])

	model.model.save(model.name)
