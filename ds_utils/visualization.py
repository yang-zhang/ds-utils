import sklearn.decomposition
import sklearn.datasets

import seaborn as sns
import matplotlib.pyplot as plt

import ds_utils.preprocessing
import ds_utils.explore


def df_hist(df, col, bins=None):
    ax = plt.axes()
    sns.distplot(df[col].dropna(), bins=bins);
    ax.set_title(col)


def df_scatterplot(df, numerical_col_1, numerical_col_2):
    sns.jointplot(x=numerical_col_1, y=numerical_col_2, data=df);


def df_barplot_frequency(df, col):
    df_freq = ds_utils.explore.df_categorical_col_value_percent(df, col)
    print df_freq
    ax = plt.axes()
    sns.barplot(x=col, y='ratio', data=df_freq)
    ax.set_title(col)


def df_boxplot_numerical_col_by_categorical_col(df, numerical_col, categorical_col):
    sns.boxplot(x=categorical_col, y=numerical_col, data=df);


def df_stackedbarplot(df, col_1, col_2):
    df_pivot = ds_utils.explore.df_describe_categorical_col_by_categorical_col(df, col_1, col_2)
    df_pivot.T.plot(kind='bar', stacked=True)


def df_pairplot(df, cols):
    sns.pairplot(df[cols].dropna());


# http://scikit-learn.org/stable/tutorial/statistical_inference/putting_together.html
def plot_pca_explained_variance_ratio(X, n_components=None):
    pca = sklearn.decomposition.PCA(n_components=n_components)
    pca.fit(X)
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_ratio_.cumsum(), linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained_variance_ratio_')


if __name__ == '__main__':
    # best way to run: in jupyter notebook, run "run ds_utils / visualization.py".
    df = ds_utils.base.make_test_df()
    df = ds_utils.preprocessing.preprocess_test_df(df)
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

    plot_pca_explained_variance_ratio(sklearn.datasets.load_digits().data)
