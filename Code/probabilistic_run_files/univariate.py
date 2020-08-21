import pandas as pd
from cross_validation_prob import ProbabilisticForecastsAnalyzer
from models_hyperparams_prob import possible_hyperparams_per_model as hyperparameters
from container import *

if __name__ == '__main__':
    KEY = 'univariate'

    path = datasets[KEY]
    print('\n=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/   {}   =/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/=/'.format(KEY))

    df_train_collated = pd.read_csv(path + 'df_train_collated.csv')
    df_test_collated = pd.read_csv(path + 'df_test_collated.csv')

    # specify output folder to save plots in
    output_folder = '../prob_results/{}/'.format(KEY)

    pfa = ProbabilisticForecastsAnalyzer(df_train_collated, target_variable='demand',
                                         split_ratio=0.2, output_folder=output_folder,
                                         scale=True,
                                         scale_output=False,
                                         output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                                         input_zscore=None, input_minmax=scaling[KEY], input_box=None, input_log=None,
                                         cols_drop=None,
                                         grid=True, random_grid=False,
                                         nb_folds_grid=5, nb_repeats_grid=None,
                                         testing_data=df_test_collated,
                                         save_errors_xlsx=True,
                                         save_validation=False)

    unc_models = ['bootstrap', 'mixture']
    for unc_model in unc_models:
        print('\n********** Results for {} **********'.format(unc_model))
        pfa.cross_validation_uncertainties(possible_hyperparams=hyperparameters[unc_model], model_name=unc_model)
        pfa.errors_to_csv()

    print('\n********** Results for NGBoost **********')
    pfa.cross_validation_grid_ngboost(possible_hyperparams=hyperparameters['ngboost'], sort_by='rmse')
    pfa.errors_to_csv()
    #
    print('\n********** Results for MCDropout **********')

    pfa.cross_validation_grid_mc_dropout(possible_hyperparams=hyperparameters['mc_dropout'], sort_by='rmse')
    pfa.errors_to_csv()

    print('\n********** Results for Deep Ensemble **********')
    pfa.cross_validation_deep_ensemble(possible_hyperparams=hyperparameters['deep_ensemble'], sort_by='rmse')
    pfa.errors_to_csv()
