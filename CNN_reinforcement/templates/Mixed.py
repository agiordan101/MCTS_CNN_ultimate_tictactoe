import tensorflow as tf
from CNN_connector import RLModel

class Mixed(RLModel):

	def __init__(self):
		super().__init__(model_path='models/mixed')
		print(self.model.name)

	def get_model(self, use_mask=True, summary=True):
		input_shape = (9, 9, 2) if use_mask else (9, 9, 1)
		i = tf.keras.layers.Input(shape=input_shape)
		c1 = tf.keras.layers.Conv2D(
			filters=128,
			kernel_size=(3,3),
			strides=(3, 3),
			padding='valid',
			activation='relu'
		)(i)

		flatt = tf.keras.layers.Flatten()(c1)

		d1 = tf.keras.layers.Dense(512, activation='relu')(flatt)
		d2 = tf.keras.layers.Dense(256, activation='relu')(d1)

		pi = tf.keras.layers.Dense(81, activation='softmax')(d2)
		v = tf.keras.layers.Dense(1, activation='tanh')(d2)

		model = tf.keras.models.Model(inputs=i, outputs=[pi, v])
		model.compile(
			loss=['categorical_crossentropy','mean_squared_error'],
			optimizer='Adam'
		)
		if summary:
			model.summary()
		return model

