import pandas as pd
from cross_validation import LearningModel
from models_hyperparams_grid import possible_hyperparams_per_model as hyperparameters, models_to_test
from container import *

# collated training data
KEY = 'all_columns_minus_weather_minus_civilians'
path = datasets[KEY]
df_train_collated = pd.read_csv(path + 'df_train_collated.csv')
df_test_collated = pd.read_csv(path + 'df_test_collated.csv')

# specify output folder to save plots in
output_folder = '../old_output/shallow_tctc/%s/train_collated_test_collated/' % KEY

lm = LearningModel(df_train_collated, target_variable='demand',
                   split_ratio=0.2, output_folder=output_folder,
                   scale=True,
                   scale_output=False,
                   output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                   input_zscore=None, input_minmax=scaling[KEY], input_box=None, input_log=None,
                   cols_drop=None,
                   grid=True, random_grid=False,
                   nb_folds_grid=10, nb_repeats_grid=10,
                   testing_data=df_test_collated,
                   save_errors_xlsx=True,
                   save_validation=False)

for model in models_to_test:
    model_name = models_to_test[model]
    print('\n********** Results for %s **********' % model_name)

    # cross validation
    lm.cross_validation(model, hyperparameters[model_name], model_name)

    # saving error metrics in a csv file
    lm.errors_to_csv()

