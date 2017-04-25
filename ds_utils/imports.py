"""Frequent imports and setups. Add the following to the beginning of code:
    import ds_utils.imports; import imp; imp.reload(ds_utils.imports)
    from ds_utils.imports import

"""

import sys
import os
from imp import reload

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.metrics

import sklearn.dummy
import sklearn.linear_model
import sklearn.ensemble

import xgboost as xgb

os.environ["KERAS_BACKEND"] = "theano"

import keras

from keras import backend as K

K.set_image_dim_ordering('th')
