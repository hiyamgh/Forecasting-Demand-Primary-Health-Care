import glob
import pandas as pd
import numpy as np
from Code.cross_validation import replace_zeros_with_ones, check_service_mohafaza
from Code.timeseries import get_timeseries_components
import warnings
warnings.filterwarnings("ignore")
import os


def collate_train_test(path, split_ratio, output_folder, drop_date=True):
    '''
    This code is responsible for collating our 12 datasubsets together. As the modeling results of each indiviual data
    subset was not in its best performnace, due to the small size of each data subset, we decided to collate
    all 12 together in order to improve modelling results in places were it was week. This is done as follows:
    1. read each data subset
    2. some data cleaning
    3. add a feature cross column that resembles the geo-location of a certain Governorate (mohafaza). This is done
    by taking the longitude and latitude of the current mohafaza, rounding both to the nearest integer, and then
    multiplying them.
    4. add the service column that resembles the service type of the current data subset
    5. after collation is done (vertical merge), we do one-hot encoding for the service, creating 4 additional columns,
     one for each service.
    6. drop the service column, because we don't need it after creating the dummy columns.
    :param path: path to the directory containing the datasubsets
    :param split_ratio: ration of the testing data. The remaining is taken for the training.
    :return: the collated dataset.
    '''
    files = glob.glob(path + '*.csv')
    training_data_frames = []
    testing_data_frames = []

    mohafaza_cities = {'Tripoli': "Zgharta-Ehden",
                       'bikaa': "Zahle",
                       'akkar': "Al-Qoubaiyat"}

    cities_rep = {
        'Zgharta-Ehden': 'N',
        'Zahle': 'B',
        'Al-Qoubaiyat': 'NE'
    }

    columns = []
    for file in files:
        df = pd.read_csv(file)

        curr_service, curr_mohafaza, curr_datasubset = check_service_mohafaza(file)

        # drop the date column
        if drop_date:
            df = df.drop('date', axis=1)

            # cols = ['demand'] + [col for col in list(df.columns.values) if col != 'demand']
            # df = df[cols]

        # since lags are either 4 or 5, we will drop the fifth lag from the data subset
        # if it happens to exist, so that we are able to merge.
        if 'w_{t-5}' not in list(df.columns.values):
            df['w_{t-5}'] = df['demand'].shift(5)

        # drop NaN values that result from lags
        df = df.dropna()

        df = get_timeseries_components(df, 'w_{t-1}', model='additive', freq=52)

        # replace zeros with ones to avoid MAPE being inf
        df = replace_zeros_with_ones(df)

        # add a column for the service
        df['service'] = str(curr_service)

        # add a column for the mohafaza (Governorate)
        df['mohafaza'] = cities_rep[mohafaza_cities.get(curr_mohafaza)]

        nb_rows_test = int(round(len(df) * split_ratio))
        nb_rows_train = len(df) - nb_rows_test

        df_train = df[0: nb_rows_train]
        df_test = df[nb_rows_train:]

        # round all columns that need rounding to the nearest integer, for both the training and the testing parts
        for col in list(df_train.columns.values):
            if df_train[col].dtype == np.float64:
                df_train[col] = df_train[col].round().astype(int)

        for col in list(df_test.columns.values):
            if df_test[col].dtype == np.float64:
                df_test[col] = df_test[col].round().astype(int)

        columns = list(df_train.columns.values)

        # append the data subset to the list
        training_data_frames.append(df_train)
        testing_data_frames.append(df_test)

        # testing_data_frames[curr_datasubset] = df_test

    # collating the training data of all data subsets together
    df_train_collated = pd.concat(training_data_frames)
    df_test_collated = pd.concat(testing_data_frames)

    df_train_collated = df_train_collated[columns]
    df_test_collated = df_test_collated[columns]

    # one hot encoding for the service variable
    df_train_collated = pd.concat([df_train_collated, pd.get_dummies(df_train_collated['service'], prefix='service')], axis=1)
    df_test_collated = pd.concat([df_test_collated, pd.get_dummies(df_test_collated['service'], prefix='service')], axis=1)

    # one hot encoding for the mohafaza variable
    df_train_collated = pd.concat([df_train_collated, pd.get_dummies(df_train_collated['mohafaza'], prefix='mohafaza')], axis=1)
    df_test_collated = pd.concat([df_test_collated, pd.get_dummies(df_test_collated['mohafaza'], prefix='mohafaza')], axis=1)

    # drop the 'service' & 'mohafaza' columns as now they are represented by vectors (one hot encoded)
    df_train_collated = df_train_collated.drop(['service', 'mohafaza'], axis=1)
    df_test_collated = df_test_collated.drop(['service', 'mohafaza'], axis=1)

    output_folder_training_collated = output_folder + 'training_collated/'
    output_folder_testing_collated = output_folder + 'testing_collated/'

    if not os.path.exists(output_folder_training_collated):
        os.makedirs(output_folder_training_collated)
    if not os.path.exists(output_folder_testing_collated):
        os.makedirs(output_folder_testing_collated)

    df_train_collated.to_csv(output_folder_training_collated + 'df_train_collated.csv', index=False)
    df_test_collated.to_csv(output_folder_testing_collated + 'df_test_collated.csv', index=False)

    return df_train_collated, df_test_collated


def save_collated_data(df_train_collated, df_test_collated, output_folder, cols_drop=None):
    # dropping columns
    if cols_drop is not None:
        df_train_collated = df_train_collated.drop(cols_drop, axis=1)
        df_test_collated = df_test_collated.drop(cols_drop, axis=1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df_train_collated.to_csv(output_folder + 'df_train_collated.csv', index=False)
    df_test_collated.to_csv(output_folder + 'df_test_collated.csv', index=False)


def save_separated_data(df, service_name, mohafaza, output_folder, cols_drop=None):
    # dropping columns
    if cols_drop is not None:
        df = df.drop(cols_drop, axis=1)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    df.to_csv(os.path.join(output_folder, '%s_%s.csv' % (service_name, mohafaza)), index=False)


def generate_train_collated_test_collated():
    input_path = '../old_stuff/output/multivariate_datasubsets/'
    output_path_without_date = '../input/collated_without_date/'
    output_path_with_date = '../input/collated_with_date/'

    # without date
    collate_train_test(input_path, 0.2, output_path_without_date, drop_date=True)

    # with date
    collate_train_test(input_path, 0.2, output_path_with_date, drop_date=False)


def generate_all_separated(output_folder, with_date=False):
    # read the training and testing collated_univariate_multivariate datasets

    output_folder = output_folder + 'separated/'
    services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
    mohafazas = ['akkar', 'bikaa', 'Tripoli']
    for service in services:
        for mohafaza in mohafazas:
            # df = pd.read_csv('../input/separated/%s_%s.csv' % (service, mohafaza))
            df = pd.read_csv('../old_stuff/output/multivariate_datasubsets/%s_%s.csv' % (service, mohafaza))

            if not with_date:
                df = df.drop(['date'], axis=1)

            # generate all columns
            save_separated_data(df, service, mohafaza, output_folder + 'all_columns/', cols_drop=None)

            # generate univariate (collated_univariate_multivariate)
            cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation',
                         'w_{t-1}_trend', 'w_{t-1}_seasonality', 'civilians_rank', 'distance']

            save_separated_data(df, service, mohafaza, output_folder + 'univariate/', cols_drop=cols_drop)

            # generate all columns minus weather (collated_univariate_multivariate)
            cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation']
            save_separated_data(df, service, mohafaza, output_folder + 'all_columns_minus_weather/', cols_drop=cols_drop)

            # generate all columns minus weather minus lags (collated_univariate_multivariate)
            if 'w_{t-5}' not in list(df.columns.values):
                cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'w_{t-1}', 'w_{t-2}', 'w_{t-3}', 'w_{t-4}', 'w_{t-1}_trend', 'w_{t-1}_seasonality']
            else:
                cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'w_{t-1}', 'w_{t-2}', 'w_{t-3}', 'w_{t-4}', 'w_{t-5}', 'w_{t-1}_trend', 'w_{t-1}_seasonality']
            save_separated_data(df, service, mohafaza, output_folder + 'all_columns_minus_weather_minus_lags/', cols_drop=cols_drop)

            # generate all columns minus weather minus vdc (distance and civilians rank)
            cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'civilians_rank', 'distance']
            save_separated_data(df, service, mohafaza, output_folder + 'all_columns_minus_weather_minus_vdc/', cols_drop=cols_drop)

            # generate all columns minus weather minus distance
            cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'distance']
            save_separated_data(df, service, mohafaza, output_folder + 'all_columns_minus_weather_minus_distance/',
                                cols_drop=cols_drop)

            # generate all columns minus weather minus civilians_rank
            cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'civilians_rank']
            save_separated_data(df, service, mohafaza, output_folder + 'all_columns_minus_weather_minus_civilians_rank/',
                                cols_drop=cols_drop)

            # generate all columns minus weather minus lags minus distance
            cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'w_{t-1}', 'w_{t-2}', 'w_{t-3}', 'w_{t-4}', 'w_{t-1}_trend', 'w_{t-1}_seasonality',
                         'distance']
            save_separated_data(df, service, mohafaza,
                                output_folder + 'all_columns_minus_weather_minus_lags_minus_distance/',
                                cols_drop=cols_drop)

            # generate all columns minus weather minus lags minus civilians rank
            cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'w_{t-1}', 'w_{t-2}', 'w_{t-3}', 'w_{t-4}',
                         'w_{t-1}_trend', 'w_{t-1}_seasonality',
                         'distance']
            save_separated_data(df, service, mohafaza,
                                output_folder + 'all_columns_minus_weather_minus_lags_minus_civilians_rank/',
                                cols_drop=cols_drop)


def generate_all_collated(output_folder, with_date=False):
    # read the training and testing collated_univariate_multivariate datasets
    if with_date:
        df_train_collated = pd.read_csv('../input/collated_with_date/training_collated/df_train_collated.csv')
        df_test_collated = pd.read_csv('../input/collated_with_date/testing_collated/df_test_collated.csv')
    else:
        df_train_collated = pd.read_csv('../input/collated_without_date/training_collated/df_train_collated.csv')
        df_test_collated = pd.read_csv('../input/collated_without_date/testing_collated/df_test_collated.csv')

    output_folder = output_folder + 'collated/'

    # generate all columns
    save_collated_data(df_train_collated, df_test_collated, output_folder + 'all_columns/', cols_drop=None)

    # generate univariate (collated_univariate_multivariate)
    cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation',
                 'w_{t-1}_trend', 'w_{t-1}_seasonality',
                 'service_General Medicine', 'service_Gynaecology', 'service_Pediatrics', 'service_Pharmacy',
                 'mohafaza_NE', 'mohafaza_N', 'mohafaza_B', 'civilians_rank', 'distance']

    save_collated_data(df_train_collated, df_test_collated, output_folder + 'univariate/', cols_drop=cols_drop)

    # generate all columns minus weather (collated_univariate_multivariate)
    cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation']
    save_collated_data(df_train_collated, df_test_collated, output_folder + 'all_columns_minus_weather/', cols_drop=cols_drop)

    # generate all columns minus weather minus lags (collated_univariate_multivariate)
    cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'w_{t-1}', 'w_{t-2}',
                 'w_{t-3}', 'w_{t-4}', 'w_{t-5}', 'w_{t-1}_trend', 'w_{t-1}_seasonality']
    save_collated_data(df_train_collated, df_test_collated, output_folder + 'all_columns_minus_weather_minus_lags/', cols_drop=cols_drop)

    # generate all columns minus weather minus vdc (distance and civilians rank)
    cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'civilians_rank', 'distance']
    save_collated_data(df_train_collated, df_test_collated, output_folder + 'all_columns_minus_weather_minus_vdc/', cols_drop=cols_drop)

    # generate all columns minus weather minus distance
    cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'distance']
    save_collated_data(df_train_collated, df_test_collated, output_folder + 'all_columns_minus_weather_minus_distance/',
                       cols_drop=cols_drop)

    # generate all columns minus weather minus civilians
    cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'civilians_rank']
    save_collated_data(df_train_collated, df_test_collated, output_folder + 'all_columns_minus_weather_minus_civilians/',
                       cols_drop=cols_drop)

    # generate all columns minus weather minus lags minus distance
    cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'w_{t-1}', 'w_{t-2}',
                 'w_{t-3}', 'w_{t-4}', 'w_{t-5}', 'w_{t-1}_trend', 'w_{t-1}_seasonality', 'distance']
    save_collated_data(df_train_collated, df_test_collated,
                       output_folder + 'all_columns_minus_weather_minus_lags_minus_distance/',
                       cols_drop=cols_drop)

    # generate all columns minus weather minus lags minus civilians
    cols_drop = ['AverageTemp', 'AverageWindSpeed', 'Precipitation', 'w_{t-1}', 'w_{t-2}',
                 'w_{t-3}', 'w_{t-4}', 'w_{t-5}', 'w_{t-1}_trend', 'w_{t-1}_seasonality', 'civilians_rank']
    save_collated_data(df_train_collated, df_test_collated,
                       output_folder + 'all_columns_minus_weather_minus_lags_minus_civilians/',
                       cols_drop=cols_drop)


if __name__ == '__main__':
    generate_train_collated_test_collated()

    outputfolder1 = '../input/all_without_date/'
    outputfolder2 = '../input/all_with_date/'

    # generate_all_collated(outputfolder1, with_date=False)
    # generate_all_collated(outputfolder2, with_date=True)

    generate_all_separated(outputfolder1, with_date=False)
    generate_all_separated(outputfolder2, with_date=True)
