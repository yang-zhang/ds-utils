# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import scipy.stats
import sklearn.metrics

import ds_utils.base
import ds_utils.preprocessing
import ds_utils.testing


def dataframize_vec_function(vec_fun):
    def df_fun(df, col_1, col_2):
        df_cols_dropna = df[[col_1, col_2]].dropna()
        return vec_fun(df_cols_dropna[col_1], df_cols_dropna[col_2])
    return df_fun


# # numerical v.s. numerical

def vec_corrcoef(x1, x2):
    corr_coef, p_value = scipy.stats.pearsonr(x1, x2)
    return {'corr_coef': corr_coef, 'p_value': p_value}


def df_cols_corrcoef(df, col_1, col_2):
    df_cols_dropna = df[[col_1, col_2]].dropna()
    return vec_corrcoef(df_cols_dropna[col_1], df_cols_dropna[col_2])


def df_corrcoef_matrix(df, numerical_cols):
    dict_dtype_col = {'float': numerical_cols}
    df = ds_utils.preprocessing.df_cast_column_types(df, dict_dtype_col)
    matrix_corrcoef = np.corrcoef(np.array(df[numerical_cols].dropna()).T)
    df_corrcoef = pd.DataFrame(matrix_corrcoef)
    df_corrcoef.columns = numerical_cols
    df_corrcoef.index = numerical_cols
    return df_corrcoef


# # categorical v.s. categorical
# ## chisq
def vec_chisq(x1, x2):
    tb = ds_utils.base.vec_table_r(x1, x2)
    return scipy.stats.chi2_contingency(tb)


def df_cols_chisq(df, col_1, col_2):
    df_cols_dropna = df[[col_1, col_2]].dropna()
    return vec_chisq(df_cols_dropna[col_1], df_cols_dropna[col_2])


# ### a refactor of the above; equivalent
def df_cols_chisq_2(df, col_1, col_2):
    df_cols_dropna = df[[col_1, col_2]].dropna()
    tb = ds_utils.base.df_table_r(df_cols_dropna, col_1, col_2)
    return scipy.stats.chi2_contingency(tb)


# # mutual info
def vec_entropy(x):
    return scipy.stats.entropy(np.bincount(x))


def vec_joint_entropy(x1, x2):
    df = pd.DataFrame(np.stack((x1, x2), axis=1))
    df.columns = ['a', 'b']
    df_value_counts_joined = df.groupby(['a', 'b']).size().reset_index().rename(columns={0: 'count'})
    value_counts_joined = df_value_counts_joined['count']
    return scipy.stats.entropy(value_counts_joined)


def vec_mutual_info(x1, x2):
    return vec_entropy(x1) + vec_entropy(x2) - vec_joint_entropy(x1, x2)


def vec_mutual_info_2(x1, x2):
    return sklearn.metrics.mutual_info_score(x1, x2)


def df_cols_mutual_info(df, col_1, col_2):
    df_cols_dropna = df[[col_1, col_2]].dropna()
    return vec_mutual_info(df_cols_dropna[col_1], df_cols_dropna[col_2])


# # numerical v.s. categorical
# ## t-test
# ### http://stackoverflow.com/questions/31768464/confidence-interval-for-t-test-difference-between-means-in-python

def vec_t_test_conf_interval(x1, x2):
    cm = sms.CompareMeans(sms.DescrStatsW(np.array(x1)), sms.DescrStatsW(np.array(x2)))
    return cm.tconfint_diff(usevar='unequal')


def vec_t_test(vec_binary, vec_num):
    binary_v1, binary_v2 = np.unique(vec_binary)
    x1 = vec_num[vec_binary == binary_v1]
    x2 = vec_num[vec_binary == binary_v2]
    t, p = scipy.stats.ttest_ind(x1, x2)
    confidence_interval = vec_t_test_conf_interval(x1, x2)
    return {
        't': t,
        'p': p,
        'confidence_interval': confidence_interval
    }


def df_cols_t_test(df, col_binary, col_num):
    """
    Use cases:
        col_binary: binary feature; col_num: binary target (e.g., A/B test on conversion)
        col_binary: binary feature; col_num: numerical target (e.g., A/B test on revenue)
        col_binary: binary target; col_num: numerical feature (e.g., age on conversion)
    """
    df_cols_dropna = df[[col_binary, col_num]].dropna()
    return vec_t_test(df_cols_dropna[col_binary], df_cols_dropna[col_num])


# ## anova
def vec_anova(vec_cat, vec_num):
    cat_col_unique_values = np.unique(vec_cat)
    list_vec_per_value = []
    for v in cat_col_unique_values:
        list_vec_per_value.append(
            vec_num[vec_cat == v]
        )
    return scipy.stats.f_oneway(*list_vec_per_value)


def df_cols_anova(df, col_cat, col_num):
    df_cols_dropna = df[[col_cat, col_num]].dropna()
    return vec_anova(df_cols_dropna[col_cat], df_cols_dropna[col_num])


class TestStatsMethods(unittest.TestCase):
    test_df = ds_utils.testing.make_test_df()
    df = ds_utils.preprocessing.df_cast_column_types(test_df, ds_utils.testing.test_df_dict_dtype_col)

    print(df.sample(5))

    n = 1000
    some_numerical = np.random.uniform(0, 1, n)
    some_numerical_with_noise = some_numerical + 0.1 * np.random.randn(n)

    def generate_random_ints(num_categories, n):
        some_random_int = np.random.randint(0, num_categories, n)
        correlated_random_int = some_random_int.copy()

        for i in range(len(some_random_int)):
            if np.random.uniform(0, 1) > 0.9:
                correlated_random_int[i] = np.random.randint(0, num_categories, 1)
        return some_random_int, correlated_random_int

    some_random_int, correlated_random_int = generate_random_ints(3, n)
    uncorrelated_random_int = generate_random_ints(3, n)[0]
    numerical_correlated_to_some_categorical = np.array([np.random.normal(c, 1) for c in some_random_int])

    some_random_binary = generate_random_ints(2, n)[0]
    numerical_correlated_to_some_binary = np.array([np.random.normal(c, 1) for c in some_random_binary])

    def test_corrcoef(self):
        print(vec_corrcoef(self.some_numerical, self.some_numerical_with_noise))
        print(df_cols_corrcoef(self.df, 'total_purchase', 'income'))
        print(df_corrcoef_matrix(self.df, numerical_cols=['total_purchase', 'income', 'tax']))

    def test_chisq(self):
        print(vec_chisq(self.some_random_int, self.correlated_random_int))
        print(vec_chisq(self.df['has_churned'], self.df['price_plan']))
        print(df_cols_chisq(self.df, 'has_churned', 'price_plan'))

    def test_mutual_info(self):
        mi1 = vec_mutual_info(self.some_random_int, self.correlated_random_int)
        mi2 = vec_mutual_info_2(self.some_random_int, self.correlated_random_int)
        self.assertAlmostEqual(mi1, mi2)

    def test_t_test(self):
        print(vec_t_test(self.some_random_binary, self.numerical_correlated_to_some_binary))
        print(df_cols_t_test(self.df, 'has_churned', 'income'))

    def test_anova(self):
        print(vec_anova(self.some_random_int, self.numerical_correlated_to_some_categorical))
        print(df_cols_anova(self.df, 'price_plan', 'total_purchase'))


if __name__ == '__main__':
    unittest.main()
