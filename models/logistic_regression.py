'''
  Logisitic Regression
'''

# libraries
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

sys.path.append('..')

# modules
import modules.gen_data as gd


# main
if __name__ == '__main__':

  X, y = gd.gen_cla(n=1000, f=4, classes=2, rs=33)
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

  classifier = LogisticRegression().fit(x_train, y_train)

  y_pred = classifier.predict(x_test)
  print(f'Logisitic regression\n'
        f'Coefficients: {classifier.coef_}\n'
        f'Intercept: {classifier.intercept_}\n'
        f'Confusion matrix:\n'
        f'{confusion_matrix(y_test, y_pred)}\n'
        f'Accuracy score: {accuracy_score(y_test, y_pred):.3f}')
  