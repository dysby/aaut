from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    Flatten,
    Input,
    Rescaling,
)
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
import tensorflow as tf
# import time
import numpy as np
from sklearn.model_selection import train_test_split
# import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, balanced_accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
# from tensorflow.python.keras.engine import input_layer

X = np.load("./data/Xtrain_Classification_Part2.npy")
y = np.load("./data/Ytrain_Classification_Part2.npy")

# X = np.reshape(X, (len(X), 50, 50, 1))
X = X/255
x_train, x_test, y_train_v, y_test_v = train_test_split(X, y, test_size=0.2)

rf = RandomForestClassifier()

rf.fit(X,y)
# Predictions on training and validation
y_hat_train = rf.predict(x_train)
    # predictions for test
y_hat_test = rf.predict(x_test)
    # training metrics
print("Training metrics:")
print(classification_report(y_true= y_train_v, y_pred=y_hat_train))
    
    # test data metrics
print("Test data metrics:")
print(classification_report(y_true= y_test_v, y_pred=y_hat_test))

# Predictions on testset
# y_pred_test = rf.predict(X_final_test)
    # test data metrics
# print("Test data metrics:")
# print(classification_report(y_true=y_final_test, y_pred= y_pred_test))

bacc_val  = balanced_accuracy_score(y_test_v, y_hat_test)
print("BACC:", bacc_val)

disp = ConfusionMatrixDisplay.from_predictions(y_test_v, y_hat_test,display_labels=["White", "Indial", "aaa", "bbbb"])
disp.plot()
plt.show()

