from wavenet import WaveNetClassifier
import numpy as np

if __name__ == '__main__':
	wnc = WaveNetClassifier((48000,), (5,), kernel_size = 2, dilation_depth = 9, n_filters = 40, task = 'classification')
	#read in datasets
	X_train = np.load("x_5000.npy")
	#Y should be 1-hot encoding! not a numerical classifier
	Y_train = np.load("y_5000.npy")
	X_val = np.load("xval_100.npy")
	Y_val = np.load("yval_100.npy")
	#single test sample
	X_test = np.load("xtest.npy")
	wnc.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs = 100, batch_size = 32, optimizer='adam', save=True, save_dir='./')
	y_pred = wnc.predict(X_test)