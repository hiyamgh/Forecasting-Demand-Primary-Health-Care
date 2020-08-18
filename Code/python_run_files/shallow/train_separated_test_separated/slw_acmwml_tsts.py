import pandas as pd
import glob
from cross_validation import LearningModel
from models_hyperparams_grid import possible_hyperparams_per_model as hyperparameters, models_to_test
import os
from container import *

KEY = 'all_columns_minus_weather_minus_lags'

# specify services and mohafazas
services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
mohafazas = ['akkar', 'bikaa', 'Tripoli']

# specify output folder for plots
output_folder = '../old_output/shallow_tsts/all_columns_minus_weather_minus_lags/train_separated_test_separated/'

# run cross validation on each separate daat subset, discarding weather columns
for file in glob.glob(datasets_separated[KEY] + '*.csv'):
    for service in services:
        for mohafaza in mohafazas:
            if service in file and mohafaza in file:
                # replace instances were demand is 0 with 1 to avoid MAPE being infinity
                df = pd.read_csv(file)
                df = df.dropna()
                df = df.replace({'demand': {0: 1}})
                print('###################### DATASUBSET: %s_%s ######################' % (service, mohafaza))
                lm = LearningModel(df, target_variable='demand',
                                    split_ratio=0.2, output_folder=output_folder + '%s_%s/' % (service, mohafaza),
                                    scale=True,
                                    scale_output=False,
                                    output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                                    input_zscore=None, input_minmax=(0, 2), input_box=None, input_log=None,
                                    cols_drop=None,
                                    grid=True, random_grid=False,
                                    nb_folds_grid=10, nb_repeats_grid=10,
                                    save_errors_xlsx=True,
                                    save_validation=False)

                for model in models_to_test:
                    model_name = models_to_test[model]
                    print('\n********** Results for %s **********' % model_name)
                    lm.cross_validation(model, hyperparameters[model_name], model_name)

                if lm.results is not None:
                    errors_df = lm.results
                    path = lm.output_folder + 'error_metrics_csv/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    errors_df.to_csv(path + 'errors.csv')

