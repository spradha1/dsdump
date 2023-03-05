'''
  Multiple Layer Perceptron/Neural Network vs. Random Forest vs. AdaBoost
'''

# libraries
from _imports import *

# global
RANDOM_STATE=96

# main
if __name__ == '__main__':

  datasets = [
    make_moons(n_samples=1000, noise=0.25, random_state=RANDOM_STATE),
    make_blobs(n_samples=1000, n_features=2, centers=[(-1, -1), (1, 1)], random_state=RANDOM_STATE),
    make_classification(n_samples=1000, n_features=2, n_clusters_per_class=2, n_redundant=0, n_informative=2, random_state=RANDOM_STATE)
  ]

  classifiers = [
    MLPClassifier(activation='logistic', alpha=0.1, max_iter=1000, random_state=RANDOM_STATE),
    RandomForestClassifier(criterion='entropy', min_impurity_decrease=0.01, random_state=RANDOM_STATE),
    AdaBoostClassifier(n_estimators=100, random_state=RANDOM_STATE)
  ]
  classifier_names = ['Multilayer Perceptron', 'Random Forest', 'AdaBoost Classifier']

  # testing
  dl, cl = len(datasets), len(classifiers)
  fig, axes = plt.subplots(dl, cl, figsize=(9, 9))
  cm = plt.cm.bwr

  for d, ds in enumerate(datasets):
    X, y = ds
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.35)

    # grid
    px = 0.05
    x1_min, x1_max = X[:, 0].min(), X[:, 0].max()
    x2_min, x2_max = X[:, 1].min(), X[:, 1].max()
    x1s = np.arange(x1_min - 0.1, x1_max + 0.1, px)
    x2s = np.arange(x2_min - 0.1, x2_max + 0.1, px)
    x1_mesh, x2_mesh = np.meshgrid(x1s, x2s)
    mesh = np.vstack( (np.reshape(x1_mesh, (1, -1)), np.reshape(x2_mesh, (1, -1))) ).T

    axes[0, d].set_title(classifier_names[d])
    axes[d, 0].set_ylabel(f'Dataset #{d+1}')

    for c, clf in enumerate(classifiers):
      ax = axes[d, c]

      # predict for whole grid
      clf_fit = clf.fit(x_train, y_train)
      probs = clf_fit.predict_proba(mesh)
      score = clf.score(x_test, y_test)
      zs = clf.predict_proba(mesh)[:, 1]
      ax.contourf(x1_mesh, x2_mesh, zs.reshape(x1_mesh.shape), cmap=cm, alpha=0.5)

      # plot dataset
      ax.scatter(x_train[:, 0][y_train == 1], x_train[:, 1][y_train == 1], c='r', marker='.', alpha=0.3)
      ax.scatter(x_train[:, 0][y_train == 0], x_train[:, 1][y_train == 0], c='b', marker='.', alpha=0.3)
      ax.scatter(x_test[:, 0][y_test == 1], x_test[:, 1][y_test == 1], c='r', marker='.')
      ax.scatter(x_test[:, 0][y_test == 0], x_test[:, 1][y_test == 0], c='b', marker='.')
      ax.text(x1_mesh.max() - .1, x2_mesh.min() + .1, f'{score:.2f}'.lstrip('0'), size=10, horizontalalignment='right')
    
  fig.suptitle('Multilayer Perceptron vs. Random Forest vs. AdaBoost Classifier with accuracy scores on the testing set')
  plt.tight_layout()
  plt.show()