import seaborn as sns
import matplotlib.pyplot as plt

import utils_yz.preprocessing
import utils_yz.explore


def df_hist(df, col, bins=None):
    ax = plt.axes()
    sns.distplot(df[col].dropna(), bins=bins);
    ax.set_title(col)


def df_scatterplot(df, numerical_col_1, numerical_col_2):
    sns.jointplot(x=numerical_col_1, y=numerical_col_2, data=df);


def df_barplot_frequency(df, col):
    df_freq = utils_yz.explore.df_categorical_col_value_percent(df, col)
    print df_freq
    ax = plt.axes()
    sns.barplot(x=col, y='ratio', data=df_freq)
    ax.set_title(col)


def df_boxplot_numerical_col_by_categorical_col(df, numerical_col, categorical_col):
    sns.boxplot(x=categorical_col, y=numerical_col, data=df);


def df_stackedbarplot(df, col_1, col_2):
    df_pivot = utils_yz.explore.df_describe_categorical_col_by_categorical_col(df, col_1, col_2)
    df_pivot.T.plot(kind='bar', stacked=True)


def df_pairplot(df, cols):
    sns.pairplot(df[cols].dropna());


if __name__ == '__main__':
    # best way to run: in jupyter notebook, run "run utils_yz / visualization.py".
    df = utils_yz.base.make_test_df()
    df = utils_yz.preprocessing.preprocess_test_df(df)
    print df.sample(5)
    print '-' * 50

    plt.figure()
    df_hist(df, 'income')

    plt.figure()
    df_scatterplot(df, 'income', 'tax')

    plt.figure()
    df_barplot_frequency(df, 'region')

    plt.figure()
    df_boxplot_numerical_col_by_categorical_col(df, 'income', 'has_churned')

    plt.figure()
    df_stackedbarplot(df, 'region', 'product_purchased')

    plt.figure()
    df_pairplot(df, ['has_churned', 'income', 'price_plan',
                     'product_purchased', 'region', 'tax', 'total_purchase'])
