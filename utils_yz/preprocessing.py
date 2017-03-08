# -*- coding: utf-8 -*-
import pandas as pd

import utils_yz.base


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


def preprocess_test_df(df):
    dict_dtype_col = {'float': ['income', 'user_id'],
                      'int': ['total_purchase', ],
                      'category': ['price_plan', 'product_purchased', 'region'],
                      'datetime': ['date_signed_on'],
                      }
    return df_cast_column_types(df, dict_dtype_col)


# replace null by string
def df_replace_nan_by_missing(df, col, by='Missing'):
    df_new = df.copy()
    indices_null = df_new[col].isnull()
    if df_new[col].dtype == 'category':
        df_new[col] = df_new[col].cat.add_categories([by])
    df_new.loc[indices_null, col] = by
    return df_new


if __name__ == '__main__':
    df = utils_yz.base.make_test_df()

    print df.sample(5)
    print df.dtypes
    print '-' * 50

    print 'preprocess_test_df'
    df = preprocess_test_df(df)
    print df.dtypes
    print '-' * 50

    print 'df_replace_nan_by_missing'
    df = df_replace_nan_by_missing(df, 'region', by='unknown')
    print df.region.cat.categories
    print df.sample(5)
    print '-' * 50
