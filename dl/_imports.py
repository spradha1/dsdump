# deep learning

# suppress warnings
import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


# analytics
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('dark', {'axes.grid' : False})