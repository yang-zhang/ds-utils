# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def make_test_df(n_row=1000):
    df = pd.DataFrame(
        {
            'user_id': list(range(n_row)),
            'has_churned': np.random.choice([True, False], n_row),
            'price_plan': np.random.choice(['A', 'B'], n_row),
            'total_purchase': 20 * np.random.rand(n_row) + 100,
            'income': 5 * np.random.rand(n_row) + 500,
            'tax': 2 * np.random.rand(n_row) + 50,
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


# R-like table function
def df_table_r(df, col_1, col_2):
    df_tmp = df.copy()
    df_tmp['_'] = 0
    tb = df_tmp.pivot_table(values=['_'], columns=col_1, index=col_2, aggfunc='count')
    return tb['_']


if __name__ == '__main__':
    df = make_test_df()
    print(df.sample(5))
    print('-' * 50)

    print('df_table_r')
    print(df_table_r(df, 'product_purchased', 'has_churned'))
    print('-' * 50)
