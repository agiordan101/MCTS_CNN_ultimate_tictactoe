import numpy as np
import tensorflow as tf
from timer import timer
# from sklearn.model_selection import train_test_split
# from numba import jit, njit

all_yxs = []
for y in range(9):
	for x in range(9):
		all_yxs.append((y, x))

@timer
def create_mask(moves):
	mask = np.zeros((9, 9))

	for s, coord in moves:
		mask[coord] = 1

	return mask

@timer
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
	state = game_np
	# print(f"state: {state}")
	return state

@timer
class RLModel:

	def __init__(self, model_path=None, name=None):
		self.name = name or 'model'
		# if name:
		# 	self.model = tf.keras.models.load_model(name)
		# else:

		if model_path:
			self.model = tf.keras.models.load_model(model_path)
		else:
			self.model = self.get_model_3d()
			# self.model = self.get_arnav_model()

	def get_model_3d(self):

		i = tf.keras.layers.Input(shape=(9, 9, 2))

		conv = tf.keras.layers.Conv2D(
			filters=16,
			strides=(3, 3),
			kernel_size=(3, 3),
			activation='linear')(i)

		flatten = tf.keras.layers.Flatten()(conv)

		dense1 = tf.keras.layers.Dense(512, activation='linear')(flatten)
		dense2 = tf.keras.layers.Dense(256, activation='linear')(dense1)

		policy = tf.keras.layers.Dense(81, activation='softmax', name='p')(dense2)
		value = tf.keras.layers.Dense(1, activation='tanh', name='v')(dense2)

		# conv = la
		model = tf.keras.Model(inputs=i, outputs=[policy, value])
		model.summary()
		model.compile(
			loss=['categorical_crossentropy','mean_squared_error'],
			optimizer='Adam',
			metrics=[['categorical_accuracy'], ['mean_squared_error']]
		)
		model.summary()
		return model

	def get_model(self):

		i = tf.keras.layers.Input(shape=(9, 9, 1))

		conv = tf.keras.layers.Conv2D(
			filters=32,
			strides=(3, 3),
			kernel_size=(3, 3),
			activation='linear')(i)

		flatten = tf.keras.layers.Flatten()(conv)

		dense1 = tf.keras.layers.Dense(512, activation='linear')(flatten)
		dense2 = tf.keras.layers.Dense(256, activation='linear')(dense1)

		policy = tf.keras.layers.Dense(81, activation='softmax', name='p')(dense2)
		value = tf.keras.layers.Dense(1, activation='tanh', name='v')(dense2)

		# conv = la
		model = tf.keras.Model(inputs=i, outputs=[policy, value])
		model.summary()
		model.compile(
			loss=['categorical_crossentropy','mean_squared_error'],
			optimizer='Adam',
			metrics=['mean_squared_error']
		)
		model.summary()
		return model
	
	def get_arnav_model(self):

		input_layer = tf.keras.layers.Input(shape=(9,9, 1), name="BoardInput")
		# reshape = tf.keras.layers.core.Reshape((9,9,1))(input_layer)
		conv_1 = tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', name='conv1')(input_layer)
		conv_2 = tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', name='conv2')(conv_1)
		conv_3 = tf.keras.layers.Conv2D(128, (3,3), padding='valid', activation='relu', name='conv3')(conv_2)

		conv_3_flat = tf.keras.layers.Flatten()(conv_3)

		dense_1 = tf.keras.layers.Dense(512, activation='relu', name='dense1')(conv_3_flat)
		dense_2 = tf.keras.layers.Dense(256, activation='relu', name='dense2')(dense_1)

		pi = tf.keras.layers.Dense(81, activation="softmax", name='pi')(dense_2)
		v = tf.keras.layers.Dense(1, activation="tanh", name='value')(dense_2)

		model = tf.keras.models.Model(inputs=input_layer, outputs=[pi, v])
		model.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=tf.keras.optimizers.Adam())

		model.summary()
		return model

	def fit(self, features, targets, epochs=200):

		rng_state = np.random.get_state()

		np.random.shuffle(features)
		for t in targets:
			np.random.set_state(rng_state)
			np.random.shuffle(targets)
		sep = int(0.2 * len(features))

		x_train = features[sep:]
		y_train = [t[sep:] for t in targets]
		x_test = features[:sep]
		y_test = [t[:sep] for t in targets]
		
		# x_train, x_test, y_train, y_test = train_test_split(
		# 	features, targets, test_size=0.2
		# )
		return self.model.fit(x_train, y_train,
			validation_data=(x_test, y_test),
			epochs=epochs,
			callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)]
		)
	
	def predict(self, features):
		return self.model.predict(features)

def parse(dataset_path):

	with open(dataset_path, 'r') as f:
		# dataset_str = f.read()
		# dataset_lst = dataset_str.split('\n')
		states = []
		qualities = []
		values = []
		for row in f.read().split('\n')[:-1]:
			srow = row.split(',')
			# print(srow)
			# if len(srow) == 163:
			states.append(np.array(srow[:162]).astype(float).reshape(9, 9, 2))
			# print(states[-1])
			# print()
			qualities.append(np.array(srow[162:-1]).astype(float))
			values.append(np.array(srow[-1]).astype(float))
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

	model_name = "models\model_night_0.mcts_rave_CNN"
	dataset_name = "datasets\dataset_night_0.mcts_rave_CNN"

	model = RLModel(name=model_name)

	features, targets = parse(dataset_name)
	print(f"features: {len(features)}")
	print(f"targets: {len(targets[1])}")

	model.fit(features, targets)

	print(np.reshape(features[:1], (9, 9, 2)))

	print(np.reshape(model.predict(features[:1])[0], (9, 9)))
	print(model.predict(features[:1])[1])

	print(np.reshape(targets[0][:1], (9, 9)))
	print(targets[1][:1])

	print(f"Diff: {np.round(targets[0][:1] - model.predict(features[:1])[0], 2)}")

	prediction = model.predict(np.zeros((1, 9, 9, 2)))
	print(f"test: {prediction}")
	print(f"Best move: {np.argmax(prediction[0])}")

	model.model.save(model.name)
