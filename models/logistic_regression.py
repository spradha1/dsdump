'''
  Logisitic Regression
'''

# libraries
from _imports import *


# main
if __name__ == '__main__':

  X, y = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=33)
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

  classifier = LogisticRegression().fit(x_train, y_train)

  y_pred = classifier.predict(x_test)
  print(f'Logisitic regression\n'
        f'Coefficients: {classifier.coef_}\n'
        f'Intercept: {classifier.intercept_}\n'
        f'Confusion matrix:\n'
        f'{confusion_matrix(y_test, y_pred)}\n'
        f'Accuracy score: {accuracy_score(y_test, y_pred):.3f}')
  