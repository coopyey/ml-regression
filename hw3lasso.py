import pandas as pd 
import numpy as np 
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

def read_data():
    data = pd.read_csv('boston_housing.data', header=None, delimiter=' ')

    # Start by splitting labels and data
    x = data.loc[:, data.columns != 13]
    y = data.loc[:, data.columns == 13]

    # Set training data
    x_tr = x.iloc[:100]
    y_tr = y.iloc[:100]

    # Set testing data
    x_te = x.iloc[101:]
    y_te = y.iloc[101:]

    # Ravel required for lasso to move from mx1 to 1xn for operations
    return (x_tr, y_tr.values.ravel(), x_te, y_te.values.ravel())

def main():
    (x_train, y_train, x_test, y_test) = read_data()

    a_vals = [0.01, 0.1, 1, 10, 100]
    lasso = LassoCV(alphas=a_vals, cv=15).fit(x_train, y_train)
    predict = lasso.predict(x_test)

    coes = lasso.coef_
    count = 0

    for length in range(0,len(coes)):
        if coes[length] == 0:
            count += 1
        length += 1

    print("Lasso average mean squared error: %.2f" % mean_squared_error(y_test, predict))
    print("Alpha value used for lasso: %.2f" % lasso.alpha_)
    print("Number of zero coefficients: %d" % count)

main()