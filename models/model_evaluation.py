'''
  Model evaluation measures
'''


# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score

sns.set_style('dark', {'axes.grid' : False})


# main
if __name__ == '__main__':
  
  # data
  df = pd.read_csv('../datasets/weatherAUS_clean.csv')

  # converting categorical to float for Linear Regression
  df[['RainToday', 'RainTomorrow']] = df[['RainToday','RainTomorrow']].replace(to_replace={'Yes': 1., 'No': 0.})

  # only continuous variables & exclude target
  X = df.loc[:, (df.columns!='RainTomorrow') & (df.dtypes == np.float64)]

  # normalize data
  X = preprocessing.scale(X)

  # target variable
  y = df.RainTomorrow.to_numpy()

  # training and testing data split
  x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)
  
  # classifer
  classifier = LogisticRegression().fit(x_train, y_train)
  
  # testing
  predictions = classifier.predict(x_test)
  print(f'Confusion Matrix:\n'
        f'{confusion_matrix(y_test, predictions)}\n'
        f'Accuracy: {accuracy_score(y_test, predictions):.3f}')
  auc = roc_auc_score(y_test, predictions)
  fpr, tpr, t = roc_curve(y_test, predictions, drop_intermediate=False)
  plt.plot([0, 1], [0, 1], '--', c='r', label='No Skill')
  plt.plot(fpr, tpr, '-', c='g', label=f'AUC={auc:.3f}')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('ROC curve, Logistic Regression classifier')
  plt.legend(loc='lower right')
  plt.show()
