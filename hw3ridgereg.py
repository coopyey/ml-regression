import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error

def read_data():
    data = pd.read_csv('boston_housing.data', header=None, delimiter=' ')

    # Start by splitting labels and data
    x = data.loc[:, data.columns != 13]
    y = data.loc[:, data.columns == 13]

    # Set training data
    x_tr = x.iloc[:20]
    y_tr = y.iloc[:20]

    # Set testing data
    x_te = x.iloc[21:]
    y_te = y.iloc[21:]

    return (x_tr, y_tr, x_te, y_te)

def main():
    (x_train, y_train, x_test, y_test) = read_data()

    reg = LinearRegression().fit(x_train, y_train)
    predict = reg.predict(x_test)

    print("Linear mean squared error: %.2f" % mean_squared_error(y_test, predict))

    a_vals = [0.01, 0.1, 1, 10, 100]
    ridge = RidgeCV(alphas=a_vals).fit(x_train, y_train)
    predict = ridge.predict(x_test)

    print("Ridge average mean squared error: %.2f" % mean_squared_error(y_test, predict))
    print("Alpha value used for ridge: %.2f" % ridge.alpha_)

main()