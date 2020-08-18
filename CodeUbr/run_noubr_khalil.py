import pandas as pd
import numpy as np
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats.stats import pearsonr, spearmanr
from scipy.spatial import distance


def get_stats(df, actual, predicted):
    def mean_absolute_percentage_error(y_true, y_pred):
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    if df.isnull().values.any():
        print('WARNING: DATA CONTAINS NAN -- THEY WILL BE DROPPED AUTOMATICALLY')
        df = df.dropna()

    if actual not in list(df.columns.values):
        raise ValueError('Column %s not in the list of columns for this data frame' % actual)
    if predicted not in list(df.columns.values):
        raise ValueError('Column %s not in the list of columns for this data frame' % predicted)

    y_test = df[actual]
    y_pred = df[predicted]
    nb_columns = len(list(df.columns.values)) - 1

    if not isinstance(y_test, list):
        y_test = list(y_test)
    if not isinstance(y_pred, list):
        y_pred = list(y_pred)

    n = len(y_test)
    r2_Score = r2_score(y_test, y_pred) # r-squared
    adjusted_r2 = 1 - ((1 - r2_Score) * (n - 1)) / (n - nb_columns - 1) # adjusted r-squared
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE
    mse_score = mean_squared_error(y_test, y_pred) # MSE
    mae_score = mean_absolute_error(y_test, y_pred) # MAE
    mape_score = mean_absolute_percentage_error(y_test, y_pred) # MAPE

    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    distance_corr = distance.correlation(y_test, y_pred)

    print('Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
          'Pearson: %.5f\nSpearman: %.5f\nDistance: %.5f' %
          (r2_Score, adjusted_r2, rmse_score, mse_score, mae_score, mape_score, pearson_corr, spearman_corr, distance_corr))

    return [r2_Score, adjusted_r2, rmse_score, mse_score, mae_score, mape_score, pearson_corr, spearman_corr, distance_corr]


maindir = '../output/khalil_nonubr/csv_datasets/'

dirs = [
    'all_columns',
    'all_columns_minus_weather',
    'all_columns_minus_weather_minus_lags',
    'all_columns_minus_weather_minus_vdc',
    'all_columns_minus_weather_minus_distance',
    'all_columns_minus_weather_minus_civilians',
    'all_columns_minus_weather_minus_lags_minus_distance',
    'all_columns_minus_weather_minus_lags_minus_civilians',
    'all_columns_univariate'
]

cols = ['r2', 'rmse', 'mse', 'mae', 'avg_demand']
results = pd.DataFrame(columns=cols)

for dir in dirs:
    df = pd.read_csv(os.path.join(maindir, dir + '.csv'))

    errors = get_stats(df, 'demand', 'predicted')

    results.loc['{}'.format(dir)] = pd.Series({
        'avg_demand': np.mean(df['demand']),
        'r2': errors[0],
        'rmse': errors[2],
        'mse': errors[3],
        'mae': errors[4],
    })

dest = '../output/results_sheets/khalil_noubr/'
if not os.path.exists(dest):
    os.makedirs(dest)
results.to_csv(os.path.join(dest, 'khalil_noubr.csv'))




