import numpy as np
from sklearn.model_selection import train_test_split
X = np.load("trainX150Ka_e_g_j_m_s.npy")
Y = np.load("trainY150Ka_e_g_j_m_s.npy")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

np.save("x_train105K_6lang", x_train)
np.save("y_train105K_6lang", y_train)
np.save("x_test22_5K_6lang", x_test)
np.save("y_test22_5K_6lang", y_test)
np.save("x_val22_5K_6lang", x_val)
np.save("y_val22_5K_6lang", y_val)