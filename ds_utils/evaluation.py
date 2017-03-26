# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp
import sklearn.metrics
import math


# https://www.kaggle.com/wiki/LogarithmicLoss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


# https://www.kaggle.com/c/avazu-ctr-prediction/discussion/10927
def logloss_scalar(p, y):
    epsilon = 1e-15
    p = min(max(p, epsilon), 1-epsilon)
    return -math.log(p) if y == 1. else -math.log(1. - p)


def mean_absolute_exp_diff(y1, y2):
    return np.abs(np.exp(y1) - np.exp(y2)).mean()


mean_absolute_exp_error = sklearn.metrics.make_scorer(mean_absolute_exp_diff, greater_is_better=False)

if __name__ == '__main__':
    y1 = np.arange(1, 5)
    y2 = np.arange(2, 6)
    y1_log = np.log(y1)
    y2_log = np.log(y2)
    print(np.isclose(sklearn.metrics.mean_absolute_error(y1, y2), mean_absolute_exp_diff(y1_log, y2_log)))
