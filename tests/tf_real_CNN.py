import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input = layers.Input(shape=(9,9), name="InputBoard")
input_reshape = layers.Reshape((9, 9, 1))(input)
print(input.shape)

conv1 = layers.Conv2D(128, (3, 3), activation='relu', name='conv1')(input_reshape)
conv2 = layers.Conv2D(128, (3, 3), activation='relu', name='conv2')(conv1)
conv3 = layers.Conv2D(128, (3, 3), activation='relu', name='conv3')(conv2)

flatten = layers.Flatten()(conv3)

fcl1 = layers.Dense(512, activation='relu', name='fullyConnectedLayer1')(flatten)
fcl2 = layers.Dense(256, activation='relu', name='fullyConnectedLayer2')(fcl1)

policy = layers.Dense(81, activation='softmax', name='policyMatrix')(fcl2)
win = layers.Dense(1, activation='tanh', name='win')(fcl2)

model = keras.Model(inputs=input, outputs=[policy, win])
model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=keras.optimizers.Adam(0.1))

model.summary()
for l in model.layers:
	print(f"Layer {l} -> {l.weights}\n")