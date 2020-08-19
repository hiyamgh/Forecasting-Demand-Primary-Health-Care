import pandas as pd
import os
import matplotlib.pyplot as plt


def add_feature_cross(df_train, df_test, output_folder, cols_drop=None):

    if cols_drop is not None:
        df_train = df_train.drop(cols_drop, axis=1)
        df_test = df_test.drop(cols_drop, axis=1)

    df_train['CivRankDist'] = df_train['civilians_rank'] * df_train['distance']
    df_test['CivRankDist'] = df_test['civilians_rank'] * df_test['distance']

    df_train = df_train.drop(['civilians_rank', 'distance'], axis=1)
    df_test = df_test.drop(['civilians_rank', 'distance'], axis=1)

    cols = list(df_train.columns.values)
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]

    df_train = df_train[cols]
    df_test = df_test[cols]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    df_train.to_csv(output_folder + 'df_train_collated.csv', index=False)
    df_test.to_csv(output_folder + 'df_test_collated.csv', index=False)

    return df_train, df_test


def produce_histogram(df, output_folder, fig_name):
    df['CivRankDist'].hist(bins=30, color='#A9C5D3',
                                 edgecolor='black', grid=False)
    plt.title('Histogram showing the distribution of the crossed column')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.xlabel('CivRankDist')
    plt.ylabel('Frequency')
    plt.savefig(output_folder + '%s.png' % fig_name)
    plt.close()


def produce_histogram_quantiles(df, quantiles, output_folder, fig_name):
    fig, ax = plt.subplots()
    df['CivRankDist'].hist(bins=30, color='#A9C5D3',
                                 edgecolor='black', grid=False)
    for quantile in quantiles:
        qvl = plt.axvline(quantile, color='r')
    ax.legend([qvl], ['Quantiles'], fontsize=10)
    ax.set_title('CivRankDist Histogram with Quantiles',
                 fontsize=12)
    ax.set_xlabel('CivRankDist', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    plt.savefig(output_folder + '%s.png' % fig_name)
    plt.close()


def rank_crossed(df_train, df_test, output_folder):
    # produce histogram showing the distribution of the crossed column
    produce_histogram(df_train, output_folder, 'df_train_collated_crossed_before_binning')
    produce_histogram(df_test, output_folder, 'df_test_collated_crossed_before_binning')

    quantile_list = [0, .25, .5, .75, 1.]
    train_quantiles = df_train['CivRankDist'].quantile(quantile_list)
    test_quantiles = df_test['CivRankDist'].quantile(quantile_list)

    # show the quantiles on the histogram
    produce_histogram_quantiles(df_train, train_quantiles, output_folder, 'df_train_collated_crossed_after_binning')
    produce_histogram_quantiles(df_test, test_quantiles, output_folder, 'df_test_collated_crossed_after_binning')

    # produce a ranking for the CivRankDist in descending order
    quantile_labels = [4, 3, 2, 1]

    # quantile_labels = ['0-25Q', '25-50Q', '50-75Q', '75-100Q']
    df_train['CrossRank'] = pd.qcut(df_train['CivRankDist'], q=quantile_list, labels=quantile_labels)
    df_test['CrossRank'] = pd.qcut(df_test['CivRankDist'], q=quantile_list, labels=quantile_labels)

    dest = '../input/all_with_date/collated_crossed_ranked/all_columns/'
    if not os.path.exists(dest):
        os.makedirs(dest)
    df_train.to_csv(dest + 'df_train_collated.csv', index=False)
    df_test.to_csv(dest + 'df_test_collated.csv', index=False)


if __name__ == '__main__':

    df_train_collated = pd.read_csv('../input/all_with_date/collated_crossed/all_columns/df_train_collated.csv')
    df_test_collated = pd.read_csv('../input/all_with_date/collated_crossed/all_columns//df_test_collated.csv')

    rank_crossed(df_train_collated, df_test_collated, '../old_stuff/eda_plots/vdc_plots/')

    # # all columns
    # training_data_path = '../input/all_without_date/collated/all_columns/df_train_collated.csv'
    # testing_data_path = '../input/all_without_date/collated/all_columns/df_test_collated.csv'
    #
    # df_train_collated = pd.read_csv(training_data_path)
    # df_test_collated = pd.read_csv(testing_data_path)
    #
    # add_feature_cross(df_train=df_train_collated, df_test=df_test_collated,
    #                   output_folder='../input/all_with_date/collated_crossed/all_columns/',
    #                   cols_drop=None)
    #
    # # all columns minus weather
    # training_data_path = '../input/all_without_date/collated/all_columns_minus_weather/df_train_collated.csv'
    # testing_data_path = '../input/all_without_date/collated/all_columns_minus_weather/df_test_collated.csv'
    #
    # df_train_collated = pd.read_csv(training_data_path)
    # df_test_collated = pd.read_csv(testing_data_path)
    #
    # add_feature_cross(df_train=df_train_collated, df_test=df_test_collated,
    #                   output_folder='../input/all_with_date/collated_crossed/all_columns_minus_weather/',
    #                   cols_drop=None)
    #
    # # all columns minus weather minus lags
    # training_data_path = '../input/all_without_date/collated/all_columns_minus_weather_minus_lags/df_train_collated.csv'
    # testing_data_path = '../input/all_without_date/collated/all_columns_minus_weather_minus_lags/df_test_collated.csv'
    #
    # df_train_collated = pd.read_csv(training_data_path)
    # df_test_collated = pd.read_csv(testing_data_path)
    #
    # add_feature_cross(df_train=df_train_collated, df_test=df_test_collated,
    #                   output_folder='../input/all_with_date/collated_crossed/all_columns_minus_weather_minus_lags/',
    #                   cols_drop=None)



