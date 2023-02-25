'''
  Gaussian discriminant analysis (Linear, covariance matrices same for all classes)
'''

# libraries
from _imports import *

# modules
import modules.gen_data as gd


# main
if __name__ == '__main__':

  X, y = gd.gen_blo(n=1000, f=2, c=[(-1, -1), (1, 1)], rs=33)
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  classifier = LinearDiscriminantAnalysis().fit(x_train, y_train)
  y_pred = classifier.predict(x_test)
  print(f'Linear discriminant analysis\n'
        f'Coefficients: {classifier.coef_}\n'
        f'Intercept: {classifier.intercept_}\n'
        f'Confusion matrix:\n'
        f'{confusion_matrix(y_test, y_pred)}\n'
        f'Accuracy score: {accuracy_score(y_test, y_pred):.3f}')

  