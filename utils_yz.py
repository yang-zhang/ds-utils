# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.stats.api as sms
import scipy.stats

# make test dataframe function
def make_test_df(n_row=1000):
    df = pd.DataFrame(
        {
            'id': range(n_row),
            'val_boolean_1_target': np.random.choice([True, False], n_row),
            'val_boolean_2': np.random.choice(['A', 'B'], n_row),
            'val_real_1_target': np.random.rand(n_row),
            'val_real_2': 5*np.random.rand(n_row) + 10,
            'val_categorical_1_target': np.random.choice(['X', 'Y', 'Z'], n_row),
            'val_categorical_2': np.random.choice(['a', 'b', 'c', 'd'], n_row),

        }
    )
    df.loc[np.random.choice(n_row, int(n_row*0.1), replace=False), 'val_real_2'] = None
    df.loc[np.random.choice(n_row, int(n_row*0.1), replace=False), 'val_boolean_2'] = None
    df.loc[np.random.choice(n_row, int(n_row*0.15), replace=False), 'val_categorical_2'] = None
    return df


# preprocessing

# replace null by string
def df_replace_nan_by_missing(df, col, by='Missing'):
    df_new = df.copy()
    indices_null = df_new[col].isnull()
    df_new.loc[indices_null, col] = by
    return df_new


# descriptive functions

def df_count_nulls(df, ratio=True):
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


def df_categorical_col_count_values(df, col, ratio=True):
    counts = df[col].value_counts(dropna=False).sort_index()
    if ratio:
        return counts / df.shape[0]
    else:
        return counts


def df_categorical_col_percent_by_categorical_col(df, categorical_col_1, categorical_col_2):
    return df.groupby(categorical_col_1).apply(lambda x: x[categorical_col_2].value_counts(dropna=False)/x.shape[0])


# R-like table function
def df_table_r(df, col_1, col_2):
    df_tmp = df.copy()
    df_tmp['_'] = 0
    return df_tmp.pivot_table(values=['_'], columns=col_1, index=col_2, aggfunc='count')


# statistics

# numerical v.s. numerical
def df_corrcoef_matrix(df, numerical_cols):
    return np.corrcoef(np.array(df[numerical_cols].dropna()).T)


# numerical v.s. numerical
def df_numerical_cols_corrcoef(df, col_1, col_2):
    df_cols_dropna = df[[col_1, col_2]].dropna()
    corr_coef, p_value = scipy.stats.pearsonr(df_cols_dropna[col_1], df_cols_dropna[col_2])
    return {'corr_coef': corr_coef, 'p_value': p_value}


# t-test
# http://stackoverflow.com/questions/31768464/confidence-interval-for-t-test-difference-between-means-in-python
def t_test_confidence_interval(x1, x2):
    cm = sms.CompareMeans(sms.DescrStatsW(np.array(x1)), sms.DescrStatsW(np.array(x2)))
    return cm.tconfint_diff(usevar='unequal')


# use cases
# col_binary: binary feature; col_num: binary target (e.g., A/B test on conversion)
# col_binary: binary feature; col_num: numerical target (e.g., A/B test on revenue)
# col_binary: binary target; col_num: numerical feature (e.g., age on conversion)
def df_t_test(df, col_binary, col_num):
    col_binary_v1, col_binary_v2 = df[col_binary].dropna().unique()
    x1 = df[col_num][df[col_binary] == col_binary_v1].dropna()
    x2 = df[col_num][df[col_binary] == col_binary_v2].dropna()
    t, p = scipy.stats.ttest_ind(x1, x2)
    confidence_interval = t_test_confidence_interval(x1, x2)
    return {
        't': t,
        'p': p,
        'confidence_interval': confidence_interval
    }


# chi-square test
def df_chi_square_test(df, col_1, col_2):
    contingency_table = df_table_r(df, col_1, col_2)
    return scipy.stats.chi2_contingency(contingency_table)


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
    print '-'*50

    print 'df_count_nulls'
    print df_count_nulls(df)
    print df_count_nulls(df, ratio=False)
    print '-'*50

    print 'df_col_is_unique_key'
    print df_col_is_unique_key(df, 'id')
    print '-'*50

    print 'df_numerical_col_percentile'
    print df_numerical_col_percentile(df, 'val_real_2')
    print '-'*50

    print 'df_numerical_col_percentile_by_categorical_col'
    print df_numerical_col_percentile_by_categorical_col(df, 'val_real_2', 'val_boolean_1_target')
    print df_numerical_col_percentile_by_categorical_col(df, 'val_real_2', 'val_categorical_2')
    print '-'*50

    print 'df_categorical_col_count_values'
    print df_categorical_col_count_values(df, 'val_categorical_2')
    print df_categorical_col_count_values(df, 'val_categorical_2', ratio=False)
    print '-'*50

    print 'df_categorical_col_percent_by_categorical_col'
    print df_categorical_col_percent_by_categorical_col(df, 'val_categorical_1_target', 'val_categorical_2')
    print df_categorical_col_percent_by_categorical_col(df, 'val_categorical_2', 'val_boolean_1_target')
    df_new = df_replace_nan_by_missing(df, 'val_categorical_2')
    print df_categorical_col_percent_by_categorical_col(df_new, 'val_categorical_2', 'val_boolean_1_target')
    print '-'*50

    print 'df_corrcoef_matrix'
    print df_corrcoef_matrix(df, numerical_cols=['val_real_1_target', 'val_real_2'])
    print 'df_numerical_cols_corrcoef'
    print df_numerical_cols_corrcoef(df, 'val_real_1_target', 'val_real_2')
    print '-'*50

    print 'df_t_test'
    print 'col_binary: binary feature; col_num: binary target (e.g., A/B test on conversion)'
    print df_t_test(df, 'val_boolean_2', 'val_boolean_1_target')
    print 'col_binary: binary target; col_num: numerical feature (e.g., age on conversion)'
    print df_t_test(df, 'val_boolean_1_target', 'val_real_2')
    print 'col_binary: binary feature; col_num: numerical target (e.g., A/B test on revenue)'
    print df_t_test(df, 'val_boolean_2', 'val_real_1_target')
    print '-'*50

    print 'df_table_r'
    print df_table_r(df, 'val_categorical_1_target', 'val_categorical_2')
    print '-'*50

    print 'df_chi_square_test'
    print df_chi_square_test(df, 'val_categorical_1_target', 'val_categorical_2')
    print '-'*50

