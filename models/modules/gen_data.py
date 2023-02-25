'''
  Generate datasets
'''

# libraries
from _imports import *


# data for regression problem
def gen_reg(n:int=100, f:int=2, noi:float=0.0, rs:int=None) -> Tuple[ArrayLike]:
  X, y = make_regression(n_samples=n, n_features=f, noise=noi, random_state=rs)
  return X, y

# data for classification problem
def gen_cla(n:int=100, f:int=4, classes:int=2, rs:int=None) -> Tuple[ArrayLike]:
  X, y = make_classification(n_samples=n, n_features=f, n_classes=classes, random_state=rs)
  return X, y

# data for classification problem with blobs
def gen_blo(n:int=100, f:int=2, c:int=2, rs:int=None) -> Tuple[ArrayLike]:
  X, y = make_blobs(n_samples=n, n_features=f, centers=c, random_state=rs)
  return X, y