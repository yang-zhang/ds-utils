# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.stats.api as sms


# make test dataframe function
def make_test_df(n_row=1000):
    df = pd.DataFrame(
        {
            'id': range(n_row),
            'val_boolean': np.random.choice([True, False], n_row),
            'val_real_1': np.random.rand(n_row),
            'val_real_2': 5*np.random.rand(n_row) + 10,
            'val_categorical_1': np.random.choice(['U', 'V', 'W', 'X', 'Y', 'Z'], n_row),
            'val_categorical_2': np.random.choice(['a', 'b', 'c', 'd'], n_row),

        }
    )
    df.loc[np.random.choice(n_row, int(n_row*0.1), replace=False), 'val_real_2'] = None
    df.loc[np.random.choice(n_row, int(n_row*0.15), replace=False), 'val_categorical_2'] = None
    return df


# preprocessing

# replace null by
def df_replace_nan_by_missing(df, col):
    df_new = df.copy()
    indices_null = df_new[col].isnull()
    df_new.loc[indices_null, col] = 'Missing'
    return df_new


# descriptive functions

def df_count_nulls(df, ratio=False):
    counts = df.apply(lambda x: x.isnull().sum())
    if ratio:
        return counts/df.shape[0]
    else:
        return counts


def df_col_is_unique_key(df, col):
    return df[col].unique().shape[0] == df.shape[0]


def df_numerical_col_percentile(df, col, percentile=np.linspace(0, 1, 5)):
    return df[col].dropna().quantile(percentile)


def df_numerical_col_percentile_by_categorical_col(df, numerical_feature, categorical_feature, percentile=np.linspace(0, 1, 5)):
    return df.groupby(categorical_feature).apply(lambda x: x[numerical_feature].dropna().quantile(percentile))


def df_categorical_col_value_counts(df, col, ratio=False):
    counts = df[col].value_counts(dropna=False).sort_index()
    if ratio:
        return counts / df.shape[0]
    else:
        return counts


def df_categorical_col_percent_by_categorical_col(df, categorical_col_1, categorical_col_2):
    return df.groupby(categorical_col_1).apply(lambda x: x[categorical_col_2].value_counts(dropna=False)/x.shape[0])


# statistics

# http://stackoverflow.com/questions/31768464/confidence-interval-for-t-test-difference-between-means-in-python
def t_test_confidence_interval(x1, x2):
    cm = sms.CompareMeans(sms.DescrStatsW(np.array(x1)), sms.DescrStatsW(np.array(x2)))
    return cm.tconfint_diff(usevar='unequal')


# metrics
# https://www.kaggle.com/wiki/LogarithmicLoss
def logloss(act, pred):
    epsilon = 1e-15
    pred = sp.maximum(epsilon, pred)
    pred = sp.minimum(1 - epsilon, pred)
    ll = sum(act * sp.log(pred) + sp.subtract(1, act) * sp.log(sp.subtract(1, pred)))
    ll = ll * -1.0 / len(act)
    return ll


if __name__ == '__main__':

    df = make_test_df()
    # test descriptive functions
    print df.sample(5)
    print '\n'

    print 'df_count_nulls'
    print df_count_nulls(df)
    print df_count_nulls(df, ratio=True)
    print '\n'

    print 'df_col_is_unique_key'
    print df_col_is_unique_key(df, 'id')
    print '\n'

    print 'df_numerical_col_percentile'
    print df_numerical_col_percentile(df, 'val_real_2')
    print '\n'

    print 'df_numerical_col_percentile_by_categorical_col'
    print df_numerical_col_percentile_by_categorical_col(df, 'val_real_2', 'val_boolean')
    print df_numerical_col_percentile_by_categorical_col(df, 'val_real_2', 'val_categorical_2')
    print '\n'

    print 'df_categorical_col_value_counts'
    print df_categorical_col_value_counts(df, 'val_categorical_2')
    print df_categorical_col_value_counts(df, 'val_categorical_2', ratio=True)
    print '\n'

    print 'df_categorical_col_percent_by_categorical_col'
    print df_categorical_col_percent_by_categorical_col(df, 'val_categorical_1', 'val_boolean')
    print df_categorical_col_percent_by_categorical_col(df, 'val_categorical_1', 'val_categorical_2')
    print df_categorical_col_percent_by_categorical_col(df, 'val_categorical_2', 'val_boolean')
    df_new = df_replace_nan_by_missing(df, 'val_categorical_2')
    print df_categorical_col_percent_by_categorical_col(df_new, 'val_categorical_2', 'val_boolean')
    print '\n'

