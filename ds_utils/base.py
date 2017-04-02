# -*- coding: utf-8 -*-
import unittest

import numpy as np
import pandas as pd


# R-like table function
def df_table_r(df, col_1, col_2):
    df_tmp = df.copy()
    df_tmp['_'] = 0
    tb = df_tmp.pivot_table(values=['_'], columns=col_1, index=col_2, aggfunc='count')
    return tb['_']


class TestBaseMethods(unittest.TestCase):
    def test_df_table_r(self):
        n = 100
        df = pd.DataFrame({
            'col_1': np.random.choice([True, False], n),
            'col_2': np.random.choice(['A', 'B', 'C'], n),
        })
        print(df.sample(3))
        print(df_table_r(df, 'col_1', 'col_2'))


if __name__ == '__main__':
    unittest.main()
