import pandas as pd
from cross_validation_smogn_prob import LearningModel
from models_hyperparams_prob import possible_hyperparams_per_model as hyperparameters
from container import *
KEY = 'all_columns_minus_weather_minus_civilians'
c = 'balance'

print('\n=============================================================================================================')
print('===================================        %s         =================================================' % KEY)
print('=============================================================================================================\n')

path = datasets[KEY]
df_train = pd.read_csv(path + 'df_train_collated.csv')
df_test = pd.read_csv(path + 'df_test_collated.csv')

t = (0.95, 'Euclidean', 'smogn', 'rel_mat1')

# combination1 contains the combinations for running experiments with relevance matrix
# t is a tuple of the following structure: (threshold - distance - method - rel_mat_name)

print('************************************* {} *************************************'.format(t))
vec = get_bool_method(t[2])
lm = LearningModel(df=df_train, target_variable='demand', split_ratio=0.2,
                    output_folder='../prob_results_ubr/%s_%s/%s_%s_%s_%s/' % (KEY, c, t[2], t[0], t[1], t[3]),

                    # parameters for the phi.control :) :)
                    rel_method='extremes' if len(t) == 3 else 'range',
                    extr_type='high', coef=1.5,
                    relevance_pts=None if len(t) == 3 else get_rel_mat(t[3]),

                    # parameters for the SmoteRegress, RandUnder, GaussNoiseRegress, DIBSRegress
                    rel="auto" if len(t) == 3 else get_rel_mat(t[3]), thr_rel=t[0], Cperc="balance",
                    # Cperc='balance',
                    k=5, repl=False, dist=t[1], p=2, pert=0.1,

                     # the over/under sampling method
                     smogn=vec[0], rand_under=vec[1], smoter=vec[2], gn=vec[3], nosmote=vec[4],

                     # scaling input and output
                     cols_drop=None, scale=True, scale_input=True, scale_output=False,
                     output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                     input_zscore=None, input_minmax=scaling[KEY], input_box=None, input_log=None,

                     testing_data=df_test,

                     grid=True, random_grid=False,
                     nb_folds_grid=5, nb_repeats_grid=None,
                     save_errors_xlsx=True, save_validation=False)

unc_models = ['bootstrap', 'mixture']
for unc_model in unc_models:
    print('\n********** Results for {} **********'.format(unc_model))
    lm.cross_validation_uncertainties(possible_hyperparams=hyperparameters[unc_model], model_name=unc_model)
    lm.errors_to_csv()

print('\n********** Results for ngboost **********')

lm.cross_validation_grid_ngboost(hyperparameters['ngboost'])
lm.errors_to_csv()

print('\n********** Results for MC Dropout **********')

lm.cross_validation_grid_mc_dropout(hyperparameters['mc_dropout'])
lm.errors_to_csv()

print('\n********** Results for Deep Ensembles **********')
lm.cross_validation_grid_deep_ensemble(hyperparameters['deep_ensemble'])
lm.errors_to_csv()

# statistics about rare values in the data before and after over sampling for each model
lm.rare_statistics_to_csv()

print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

