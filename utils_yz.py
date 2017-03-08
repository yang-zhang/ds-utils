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
            'user_id': range(n_row),
            'has_churned': np.random.choice([True, False], n_row),
            'price_plan': np.random.choice(['A', 'B'], n_row),
            'total_purchase': np.random.rand(n_row),
            'income': 5 * np.random.rand(n_row) + 10,
            'product_purchased': np.random.choice(['X', 'Y', 'Z'], n_row),
            'region': np.random.choice(['a', 'b', 'c', 'd'], n_row),
            'date_signed_on': np.random.choice(pd.date_range('2010-01-01', '2016-12-31'), n_row, replace=True).astype(
                'str'),

        }
    )
    df.loc[np.random.choice(n_row, int(n_row * 0.1), replace=False), 'income'] = None
    df.loc[np.random.choice(n_row, int(n_row * 0.1), replace=False), 'price_plan'] = None
    df.loc[np.random.choice(n_row, int(n_row * 0.15), replace=False), 'region'] = None
    return df


# preprocessing

def df_cast_column_types(df, dict_dtype_col):
    df_new = df.copy()
    for dtype, cols in dict_dtype_col.items():
        if dtype == 'datetime':
            for col in cols:
                df_new[col] = pd.to_datetime(df_new[col])
        else:
            for col in cols:
                df_new[col] = df_new[col].astype(dtype)
    return df_new


# replace null by string
def df_replace_nan_by_missing(df, col, by='Missing'):
    df_new = df.copy()
    indices_null = df_new[col].isnull()
    df_new.loc[indices_null, col] = by
    return df_new


# descriptive functions

def df_col_is_unique_key(df, col):
    return df[col].unique().shape[0] == df.shape[0]


def describe_numerical(x, percentiles=np.linspace(0, 1, 5)):
    d1 = {
        'count': x.shape[0],
        'mean': x.mean(),
        'sd': x.std(),
        'percent_nulls': x.isnull().sum() / float(x.shape[0]),
    }
    s1 = pd.Series(d1)
    s2 = x.dropna().quantile(percentiles)
    return pd.concat([s1, s2])


def df_describe_numerical_cols(df, cols):
    return df[cols].apply(describe_numerical)


def df_describe_numerical_cols_by_categorical_col(df, numerical_cols, categorical_col,
                                                  percentiles=np.linspace(0, 1, 5)):
    return df.groupby(categorical_col)[numerical_cols].apply(lambda x: x.dropna().quantile(percentiles))


def describe_categorical(x):
    return pd.Series(
        {
            'count': x.shape[0],
            'num_unique_value': x.nunique(),
            'percent_nulls': x.isnull().sum() / float(x.shape[0]),
        }
    )


def df_describe_categorical_cols(df, cols):
    return df[cols].apply(describe_categorical)


def df_categorical_col_value_percent(df, col):
    return df[col].value_counts(dropna=False).sort_values(ascending=False) / df.shape[0]


# R-like table function
def df_table_r(df, col_1, col_2):
    df_tmp = df.copy()
    df_tmp['_'] = 0
    tb = df_tmp.pivot_table(values=['_'], columns=col_1, index=col_2, aggfunc='count')
    return tb['_']


def df_describe_categorical_col_by_categorical_col(df, col_1, col_2):
    tb = df_table_r(df, col_1, col_2)
    return tb / tb.sum(axis=0)


# statistics

# numerical v.s. numerical
def df_corrcoef_matrix(df, numerical_cols):
    dict_dtype_col = {'float': numerical_cols}
    df = df_cast_column_types(df, dict_dtype_col)
    matrix_corrcoef = np.corrcoef(np.array(df[numerical_cols].dropna()).T)
    df_corrcoef = pd.DataFrame(matrix_corrcoef)
    df_corrcoef.columns = numerical_cols
    df_corrcoef.index = numerical_cols
    return df_corrcoef


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


# anova
def df_anova(df, col_num, col_cat):
    cat_col_unique_values = df[col_cat].dropna().unique()
    list_vec_per_value = []
    for v in cat_col_unique_values:
        list_vec_per_value.append(
            df[col_num][df[col_cat] == v].dropna()
        )
    return scipy.stats.f_oneway(*list_vec_per_value)


# evaluation
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

    print df.sample(5)
    print '-' * 50

    print 'df_cast_column_types'
    dict_dtype_col = {'float': ['income', 'user_id'],
                      'int': ['total_purchase', ],
                      'category': ['price_plan', 'product_purchased', 'region'],
                      'datetime': ['date_signed_on'],
                      }
    print df_cast_column_types(df, dict_dtype_col).dtypes
    print '-' * 50

    print 'df_replace_nan_by_missing'
    df_new = df_replace_nan_by_missing(df, 'region')
    print df_new.sample(5)
    print '-' * 50

    print 'df_col_is_unique_key'
    print df_col_is_unique_key(df, 'user_id')
    print '-' * 50

    print 'df_describe_numerical_cols'
    print df_describe_numerical_cols(df, ['total_purchase', 'income'])
    print 'df_describe_categorical_cols'
    print df_describe_categorical_cols(df, ['product_purchased', 'region'])
    print '-' * 50

    # print 'df_numerical_col_percentile_by_categorical_col'
    # print df_numerical_col_percentile_by_categorical_col(df, 'income', 'has_churned')
    # print df_numerical_col_percentile_by_categorical_col(df, 'income', 'region')
    # print '-' * 50

    print 'df_categorical_col_value_percent'
    print df_categorical_col_value_percent(df, 'region')
    print '-' * 50

    print 'df_describe_categorical_col_by_categorical_col'
    print df_describe_categorical_col_by_categorical_col(df, 'region', 'product_purchased')
    print df_describe_categorical_col_by_categorical_col(df, 'region', 'has_churned')
    df_new = df_replace_nan_by_missing(df, 'region')
    print df_describe_categorical_col_by_categorical_col(df_new, 'region', 'has_churned')
    print '-' * 50

    print 'df_corrcoef_matrix'
    print df_corrcoef_matrix(df, numerical_cols=['total_purchase', 'income'])
    print 'df_numerical_cols_corrcoef'
    print df_numerical_cols_corrcoef(df, 'total_purchase', 'income')
    print '-' * 50

    print 'df_t_test'
    print 'col_binary: binary feature; col_num: binary target (e.g., A/B test on conversion)'
    print df_t_test(df, 'price_plan', 'has_churned')
    print 'col_binary: binary target; col_num: numerical feature (e.g., age on conversion)'
    print df_t_test(df, 'has_churned', 'income')
    print 'col_binary: binary feature; col_num: numerical target (e.g., A/B test on revenue)'
    print df_t_test(df, 'price_plan', 'total_purchase')
    print '-' * 50

    print 'df_table_r'
    print df_table_r(df, 'product_purchased', 'region')
    print '-' * 50

    print 'df_chi_square_test'
    print df_chi_square_test(df, 'product_purchased', 'region')
    print '-' * 50

    print 'df_anova'
    print df_anova(df, 'total_purchase', 'region')
    print '-' * 50
