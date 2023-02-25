# special
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification, make_blobs
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
  confusion_matrix,
  accuracy_score,
  roc_auc_score,
  roc_curve,
  mean_squared_error
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# type hinting
from typing import Tuple
from numpy.typing import ArrayLike

# misc
import sys
