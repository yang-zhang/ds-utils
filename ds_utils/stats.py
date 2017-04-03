# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import scipy.stats

import ds_utils.base
import ds_utils.preprocessing
import ds_utils.testing


# numerical v.s. numerical

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


# categorical v.s. categorical
# chisq
def vec_chisq(x1, x2):
    tb = ds_utils.base.vec_table_r(x1, x2)
    return scipy.stats.chi2_contingency(tb)


def df_cols_chisq(df, col_1, col_2):
    return vec_chisq(df[col_1], df[col_2])


# a refactor of the above
def df_cols_chisq_2(df, col_1, col_2):
    tb = ds_utils.base.df_table_r(df, col_1, col_2)
    return scipy.stats.chi2_contingency(tb)


# mutual info todo
def vec_mutual_info(x1, x2):
    pass


def df_cols_mutual_info(df, col_1, col_2):
    return vec_mutual_info(df[col_1], df[col_2])


# numerical v.s. categorical
# t-test
# http://stackoverflow.com/questions/31768464/confidence-interval-for-t-test-difference-between-means-in-python
def vec_t_tet(x1, x2):
    pass


def vec_t_test_conf_interval(x1, x2):
    cm = sms.CompareMeans(sms.DescrStatsW(np.array(x1)), sms.DescrStatsW(np.array(x2)))
    return cm.tconfint_diff(usevar='unequal')


def df_t_test(df, col_binary, col_num):
    """
    Use cases:
        col_binary: binary feature; col_num: binary target (e.g., A/B test on conversion)
        col_binary: binary feature; col_num: numerical target (e.g., A/B test on revenue)
        col_binary: binary target; col_num: numerical feature (e.g., age on conversion)
    """
    col_binary_v1, col_binary_v2 = df[col_binary].dropna().unique()
    x1 = df[col_num][df[col_binary] == col_binary_v1].dropna()
    x2 = df[col_num][df[col_binary] == col_binary_v2].dropna()
    t, p = scipy.stats.ttest_ind(x1, x2)
    confidence_interval = vec_t_test_conf_interval(x1, x2)
    return {
        't': t,
        'p': p,
        'confidence_interval': confidence_interval
    }


# anova
def vec_anova(x1, x2):
    pass


def df_anova(df, col_num, col_cat):
    cat_col_unique_values = df[col_cat].dropna().unique()
    list_vec_per_value = []
    for v in cat_col_unique_values:
        list_vec_per_value.append(
            df[col_num][df[col_cat] == v].dropna()
        )
    return scipy.stats.f_oneway(*list_vec_per_value)


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

    def test_corrcoef(self):
        print(vec_corrcoef(self.some_numerical, self.some_numerical_with_noise))
        print(df_cols_corrcoef(self.df, 'total_purchase', 'income'))
        print(df_corrcoef_matrix(self.df, numerical_cols=['total_purchase', 'income', 'tax']))

    def test_chisq(self):
        print(self.some_random_int)
        print(self.correlated_random_int)
        print(vec_chisq(self.some_random_int, self.correlated_random_int))
        print(vec_chisq(self.df['has_churned'], self.df['price_plan']))
        print(df_cols_chisq(self.df, 'has_churned', 'price_plan'))

    def test_mutual_info(self):
        pass

    def test_t_test(self):
        pass

    def test_anova(self):
        pass


if __name__ == '__main__':
    unittest.main()

    # print 'df_corrcoef_matrix'
    # print df_corrcoef_matrix(df, numerical_cols=['total_purchase', 'income'])
    # print 'df_numerical_cols_corrcoef'
    # print df_numerical_cols_corrcoef(df, 'total_purchase', 'income')
    # print '-' * 50
    #
    # print 'df_t_test'
    # print 'col_binary: binary feature; col_num: binary target (e.g., A/B test on conversion)'
    # print df_t_test(df, 'price_plan', 'has_churned')
    # print 'col_binary: binary target; col_num: numerical feature (e.g., age on conversion)'
    # print df_t_test(df, 'has_churned', 'income')
    # print 'col_binary: binary feature; col_num: numerical target (e.g., A/B test on revenue)'
    # print df_t_test(df, 'price_plan', 'total_purchase')
    # print '-' * 50
    #
    # print 'df_chi_square_test'
    # print df_chi_square_test(df, 'product_purchased', 'region')
    # print '-' * 50
    #
    # print 'df_anova'
    # print df_anova(df, 'total_purchase', 'region')
    # print '-' * 50
