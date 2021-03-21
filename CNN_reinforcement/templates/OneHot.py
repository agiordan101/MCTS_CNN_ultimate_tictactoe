import tensorflow as tf
from CNN_connector import RLModel

class OneHot(RLModel):

	def __init__(self, model_path=None):
		super().__init__(model_path=model_path, name="one_hot")
		print(self.model.name)

	def get_model(self, use_mask=True, summary=True):
		input_shape = (9, 9, 3) if use_mask else (9, 9, 2)
		i = tf.keras.layers.Input(shape=input_shape)
		c1 = tf.keras.layers.Conv2D(
			filters=128,
			kernel_size=(3,3),
			strides=(3, 3),
			padding='valid',
			activation='relu',
			name='conv_layer'
		)(i)
		# c2 = tf.keras.layers.Conv2D(
		# 	filters=128,
		# 	kernel_size=(3,3),
		# 	# strides=(3, 3),
		# 	padding='valid',
		# 	activation='relu',
		# 	name='conv_layer2'
		# )(c1)
		# c2 = tf.keras.layers.Conv2D(
		# 	filters=128,
		# 	kernel_size=(3,3),
		# 	padding='valid',
		# 	activation='relu',
		# 	name='conv_layer2'
		# )(c1)
		# c3 = tf.keras.layers.Conv2D(
		# 	filters=128,
		# 	kernel_size=(3,3),
		# 	padding='valid',
		# 	activation='relu',
		# 	name='conv_layer3'
		# )(c2)

		flatt = tf.keras.layers.Flatten()(c1)

		d1 = tf.keras.layers.Dense(512, activation='relu', name='dense1')(flatt)
		d2 = tf.keras.layers.Dense(256, activation='relu', name='dense2')(d1)

		pi = tf.keras.layers.Dense(81, activation='softmax', name='policies')(d2)
		v = tf.keras.layers.Dense(1, activation='tanh', name='values')(d2)

		model = tf.keras.models.Model(inputs=i, outputs=[pi, v])
		model.compile(
			loss=['mean_squared_error'],
			optimizer='Adam'
			# metrics=['accuracy']
		)
		if summary:
			model.summary()
		# quit()
		return model

