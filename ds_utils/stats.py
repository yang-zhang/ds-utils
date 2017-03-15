# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import statsmodels.stats.api as sms
import scipy.stats

import utils_yz.base
import utils_yz.preprocessing


# TODO: clean up

# numerical v.s. numerical
def df_corrcoef_matrix(df, numerical_cols):
    dict_dtype_col = {'float': numerical_cols}
    df = utils_yz.preprocessing.df_cast_column_types(df, dict_dtype_col)
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
    contingency_table = utils_yz.base.df_table_r(df, col_1, col_2)
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


if __name__ == '__main__':
    df = utils_yz.base.make_test_df()
    df = utils_yz.preprocessing.preprocess_test_df(df)
    print df.sample(5)
    print '-' * 50

    # TODO: test
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
