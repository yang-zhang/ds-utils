# -*- coding: utf-8 -*-
import numpy as np
import scipy as sp


def df_count_nulls(df):
    return df.apply(lambda (x): x.isnull().sum())


def df_col_is_unique(df, col):
    return df[col].unique().shape[0] == df.shape[0]


def df_calculate_col_percentile(df, col, percentile=np.linspace(0, 1, 5)):
    return df[col].dropna().quantile(percentile)


def df_calculate_col_percentile_per_target_value(df, col, target, percentile=np.linspace(0, 1, 5)):
    return df.groupby(target).apply(lambda (x): x[col].dropna().quantile(percentile))


def df_calculate_col_value_counts(df, col, ratio=True):
    counts = df[col].value_counts().sort_index()
    if ratio:
        return counts / counts.sum()
    else:
        return counts


def df_calculate_target_rate_per_group(df, col, target):
    return df.groupby(col).apply(lambda (x): x[target].mean())


# metrics
# https://www.kaggle.com/wiki/LogarithmicLoss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll
