import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = layers.Input(shape=(3, 9), name="InputBoard")
input_reshape = layers.Reshape((3, 9, 1))(input)

conv1 = layers.Conv2D(16, (3, 3), activation='relu', name='conv1')(input_reshape)

flatten = layers.Flatten()(conv1)

fcl1 = layers.Dense(64, activation='relu', name='fullyConnectedLayer1')(flatten)
fcl2 = layers.Dense(32, activation='relu', name='fullyConnectedLayer2')(fcl1)

policy = layers.Dense(9, activation='softmax', name='policyMatrix')(fcl2)
win = layers.Dense(1, activation='tanh', name='win')(fcl2)

model = keras.Model(inputs=input, outputs=[policy, win])
model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=keras.optimizers.Adam(0.1))

model.summary()
for l in model.layers:
	print(f"Layer {l} -> {l.weights}\n")

	with open("./CNN_weights.data", 'a') as f:
		f.write(str(l.weights))