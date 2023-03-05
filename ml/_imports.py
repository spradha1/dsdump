# special
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.datasets import (
  make_regression,
  make_classification,
  make_blobs,
  make_circles,
  make_moons
)
from sklearn.metrics import (
  confusion_matrix,
  accuracy_score,
  roc_auc_score,
  roc_curve,
  mean_squared_error
)

# type hinting
from typing import Tuple
from numpy.typing import ArrayLike

# misc
import sys
