
from templates.OneHot import OneHot
from CNN_connector import RLModel
import numpy as np

def parse_onehot(maskon=True):

	with open("dataset_onehot.csv", 'r') as f:

		player = []
		oppponnent = []
		masks = []
		qualities = []
		values = []
		for row in f.read().split('\n')[:-1]:
			srow = row.split(',')
			# print(f"Len{len(srow)}")
			player.append(np.array(srow[:81]).astype(float).reshape(9, 9))
			oppponnent.append(np.array(srow[81:162]).astype(float).reshape(9, 9))
			masks.append(np.array(srow[162:243]).astype(float).reshape(9, 9))
			qualities.append(np.array(srow[243:-1]).astype(float))
			values.append(np.array(srow[-1]).astype(float))

		features = np.stack([np.array(player), np.array(oppponnent), np.array(masks)], axis=-1)
		print(features.shape)
		return features, [np.array(qualities), np.array(values)]


if __name__ == '__main__':

	# model_name = "models/onehot_first_test"

	model = OneHot()
	# quit()
	features, targets = parse_onehot()
	print(f"features: {len(features)}")
	print(f"targets: {len(targets[1])}")
	# quit()
	model.fit(features, targets)

	# print(np.reshape(features[:1], (9, 9, 3)))

	# print(np.reshape(model.predict(features[:1])[0], (9, 9)))
	for i in range(5):

		print(model.predict(features[i:i + 1])[1])
		print(model.predict(features[i:i + 1])[0])

		# print(np.reshape(targets[0][:1], (9, 9)))
		print(targets[0][i:i+1])
		print(targets[1][i:i+1])

	# print(f"Diff: {np.round(targets[0][:1] - model.predict(features[:1])[0], 2)}")

	# prediction = model.predict(np.zeros((1, 9, 9, 2)))
	# print(f"test: {prediction}")
	# print(f"Best move: {np.argmax(prediction[0])}")

	model.model.save("one_hot")
