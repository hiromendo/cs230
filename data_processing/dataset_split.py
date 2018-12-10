import numpy as np
from sklearn.model_selection import train_test_split
X = np.load("trainX50Karabic_german.npy")
Y = np.load("trainY50K_arabic_german.npy")
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)

np.save("x_train35K_arabic_german", x_train)
np.save("y_train35K_arabic_german", y_train)
np.save("x_test7_5K_arabic_german", x_test)
np.save("y_test7_5K_arabic_german", y_test)
np.save("x_val7_5K_arabic_german", x_val)
np.save("y_val7_5K_arabic_german", y_val)
