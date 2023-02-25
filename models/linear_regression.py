'''
  Linear Regression
'''

# libraries
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# main
if __name__ == '__main__':

  df = pd.read_csv('../datasets/weatherAUS_clean.csv')

  # target variable 'RISK_MM', keep only continuous variables
  X = df.loc[:, (df.dtypes == np.float64) & (df.columns != 'RISK_MM')].to_numpy()

  # normalize data
  X = preprocessing.scale(X)

  y = df.RISK_MM.to_numpy()
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

  classifier = LinearRegression().fit(x_train, y_train)

  y_pred = classifier.predict(x_test)
  print(f'Linear regression\n'
        f'Coefficients: {classifier.coef_}\n'
        f'Intercept: {classifier.intercept_}\n'
        f'Mean squared error: {mean_squared_error(y_test, y_pred):.3f}')