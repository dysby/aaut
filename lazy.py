from lazypredict.Supervised import LazyRegressor
import numpy as np
import pandas as pd

x_train = np.load("Xtrain_Regression_Part2.npy")
y_train = np.load("Ytrain_Regression_Part2.npy")

# x = np.array(data.drop(["Sales"], 1))
# y = np.array(data["Sales"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

reg = LazyRegressor(verbose=0, 
                    ignore_warnings=False, 
                    custom_metric=None)
models, predictions = reg.fit(xtrain, xtest, ytrain, ytest)
print(models)
