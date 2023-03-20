'''
  Cross-validation
  -> Classification case on SVM to find best regularization parameter
  -> Regression case with Ridge regression to find best alpha for optimization problem min( ||y - Xw||^2 + alpha * ||w||^2 )
'''

# libraries
from _imports import *


# global
RANDOM_STATE=44


# main
if __name__ == '__main__':

  folds = 10

  ################# classification case #####################

  X, y = datasets.load_iris(return_X_y=True)
  accuracy_scores = {}
  Cs = np.arange(0.01, 0.5, 0.03)

  # running through regularization parameters for SVM
  print(f"Cross-validation for classification with SVM to get best regularization parameter:\n\n"
    f" {'C':7}| {'Mean accuracy':13} |\n"
    f"-------------------------\n"
  )

  for c in Cs:
    clf = svm.SVC(kernel='linear', C=c, probability=True, random_state=RANDOM_STATE)
    accuracies = cross_val_score(estimator=clf, X=X, y=y, scoring='accuracy', cv=folds)
    mean_accuracy = sum(accuracies)/folds
    accuracy_scores[c] = mean_accuracy
    print(f"{round(c, 2):7} | {round(mean_accuracy, 3):13} |\n")
  
  bestC = max(accuracy_scores, key=accuracy_scores.get)
  print(f"Best C: {bestC}, Accuracy: {round(accuracy_scores[bestC], 3)}\n")


  ################# regression case #####################

  X, y = datasets.load_diabetes(return_X_y=True)
  RMSEs = {}
  alphas = [pow(10, -i) for i in range(6)]
  kfold = KFold(n_splits=folds)

  # running through alphas for Ridge regression
  print(f"Cross-validation for Ridge regression to get best alpha by root-mean-square error:\n\n"
    f" \u03B1{'':6}| {'RMSE':7} |\n"
    f"-------------------\n"
  )

  for a in alphas:
    clf = Ridge(alpha=a, max_iter=500, random_state=RANDOM_STATE)
    rmses = []

    for train_idxs, val_idxs in kfold.split(X):
      X_train, y_train = X[train_idxs], y[train_idxs]
      X_test, y_test = X[val_idxs], y[val_idxs]
      clf.fit(X_train, y_train)
      rmses.append(mean_squared_error(y_test, clf.predict(X_test), squared=False))

    RMSEs[a] = sum(rmses)/len(rmses)
    print(f"{a:7} | {round(RMSEs[a], 3):7} |\n")
  
  best_alpha = min(RMSEs, key=RMSEs.get)
  print(f"Best alpha: {best_alpha}, RMSE: {round(RMSEs[best_alpha], 3)}\n")
