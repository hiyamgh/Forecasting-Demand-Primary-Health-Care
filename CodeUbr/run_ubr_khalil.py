from utility_based_error_metrics import *
from container import *
import pandas as pd
import numpy as np
import os


maindirs = ['../output/khalil_ubr/']

dirs = {
    'all_columns': 'smogn_0.95_Euclidean',
    'all_columns_minus_weather': 'smogn_0.95_Manhattan_rel_mat1',
    'all_columns_minus_weather_minus_lags': 'smogn_0.95_Manhattan_rel_mat2',
    'all_columns_minus_weather_minus_vdc': 'smogn_0.95_Manhattan_rel_mat2',
    'all_columns_minus_weather_minus_distance': 'smogn_0.95_Euclidean_rel_mat2',
    'all_columns_minus_weather_minus_civilians': 'smogn_0.95_Euclidean_rel_mat1',
    'all_columns_minus_weather_minus_lags_minus_distance': 'smogn_0.9_Manhattan',
    'all_columns_minus_weather_minus_lags_minus_civilians': 'smogn_0.95_Manhattan_rel_mat1',
    'univariate': 'smogn_0.95_Euclidean_rel_mat2'
}

for maindir in maindirs:
    # once on khalil_ubr, once on khalil_nonubr
    cols = ['setting', 'threshold', 'distance', 'relmatrix', 'avg_demand', 'r2', 'rmse',
            'F1', 'F2', 'F05', 'prec', 'rec']
    results = pd.DataFrame(columns=cols)

    for dir in dirs:
        df_train = pd.read_csv(datasets[dir] + 'df_train_collated.csv')
        df_test = pd.read_csv(datasets[dir] + 'df_test_collated.csv')

        if '_ubr' in maindir:
            mainpath = os.path.join(maindir, 'csv_datasets')
            print(os.path.join(mainpath, dir + '_bal.csv'))
            khalilresult = pd.read_csv(os.path.join(mainpath, dir + '_bal.csv'))
            print(khalilresult.head(5))

            actual = khalilresult['demand']
            predicted = khalilresult['predicted']

            setting_orig = dirs[dir]
            setting = setting_orig.split('_')
            if len(setting) == 3:
                threshold = float(setting[1])
                distance = setting[2]
                relmat = None
                meth = 'extremes'
            else:
                threshold = float(setting[1])
                distance = setting[2]
                relmat = relmat_dict[setting[3] + '_' + setting[4]]
                meth = 'range'
            
            errors, ubrmetrics = evaluate(actual=actual, predicted=predicted, thresh=threshold,
                                          target_variable='demand',
                                          df=None, df_train=df_train, df_test=df_test,
                                          method=meth, extr_type='high', coef=1.5, control_pts=relmat)

            results.loc['{}'.format(dir)] = pd.Series({
                'setting': setting_orig,
                'threshold': threshold,
                'distance': distance,
                'relmatrix': 'None' if relmat is None else setting[3] + '_' + setting[4],
                'avg_demand': np.mean(actual),
                'r2': errors[0],
                'rmse': errors[2],
                'F1': ubrmetrics['ubaF1'][0],
                'F2': ubrmetrics['ubaF2'][0],
                'F05': ubrmetrics['ubaF05'][0],
                'prec': ubrmetrics['ubaprec'][0],
                'rec': ubrmetrics['ubarec'][0],
            })

    if '_ubr' in maindir:
        dest = '../output/results_sheets/khalil_ubr/'
        if not os.path.exists(dest):
            os.makedirs(dest)
        results.to_csv(os.path.join(dest, 'khalil_ubr.csv'))
            







