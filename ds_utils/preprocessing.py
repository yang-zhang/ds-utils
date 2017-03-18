# -*- coding: utf-8 -*-
import pandas as pd

import ds_utils.base


def df_cast_column_types(df, dict_dtype_col):
    df_new = df.copy()
    for dtype, cols in list(dict_dtype_col.items()):
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


def get_dummies_train_test(df_train, df_test, cat_cols=None):
    df_train_test = pd.concat([df_train, df_test], join='inner', keys=['train', 'test'])
    df_train_test_dummies = pd.get_dummies(df_train_test, columns=cat_cols)
    df_train_dummies = df_train_test_dummies.ix['train']
    df_test_dummies = df_train_test_dummies.ix['test']
    return df_train_dummies, df_test_dummies


def label_encode(df, cat_cols=None):
    if not cat_cols:
        cat_cols = df.columns
    df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category').cat.codes)
    return df


def label_encode_train_test(df_train, df_test, cat_cols=None):
    df_train_test = pd.concat([df_train, df_test], join='inner', keys=['train', 'test'])
    if not cat_cols:
        cat_cols = df_train_test.columns
    df_train_test[cat_cols] = df_train_test[cat_cols].apply(lambda x: x.astype('category').cat.codes)
    df_train_label_encoded = df_train_test.ix['train']
    df_test_label_encoded = df_train_test.ix['test']
    return df_train_label_encoded, df_test_label_encoded


if __name__ == '__main__':
    df = ds_utils.base.make_test_df()

    print(df.sample(5))
    print(df.dtypes)
    print('-' * 50)

    print('preprocess_test_df')
    df = preprocess_test_df(df)
    print(df.dtypes)
    print('-' * 50)

    print('df_replace_nan_by_missing')
    df = df_replace_nan_by_missing(df, 'region', by='unknown')
    print(df.region.cat.categories)
    print(df.sample(5))
    print('-' * 50)

    df_train = pd.DataFrame({
        'letter': [
            'a',
            'b',
            'c',
        ],
        'animal': [
            'dog',
            'cat',
            'dog',
        ],
        'color': [
            'red',
            'green',
            'blue',
        ],
        'number': [1., 2.5, 3.],
        'target': [True, False, True]
    })

    df_test = pd.DataFrame({
        'letter': [
            'a',
            'b',
        ],
        'animal': [
            'dog',
            'pig',
        ],
        'color': [
            'red',
            'green',
        ],
        'number': [2.5, 3.5],

    })
    print(df_train)
    print(df_test)
    categorical_cols = ['letter', 'animal', 'color']

    print(pd.get_dummies(df_train, columns=categorical_cols))

    df_train_dummies, df_test_dummies = get_dummies_train_test(
        df_train, df_test, cat_cols=categorical_cols)
    print(df_train_dummies)
    print(df_test_dummies)

    print(label_encode(df_train, cat_cols=categorical_cols))
    df_train_label_encoded, df_test_label_encoded = label_encode_train_test(
        df_train, df_test, cat_cols=categorical_cols)
    print(df_train_label_encoded)
    print(df_test_label_encoded)
