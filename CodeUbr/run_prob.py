import pandas as pd
from cross_validation_smogn_prob import LearningModel
from models_hyperparams_prob import possible_hyperparams_per_model as hyperparameters
from container import *

KEY = 'all_columns'
c = 'balance'
mat = get_rel_mat('rel_mat2')

print('\n=============================================================================================================')
print('===================================        %s         =================================================' % KEY)
print('=============================================================================================================\n')

path = datasets[KEY]
df_train = pd.read_csv(path + 'df_train_collated.csv')
df_test = pd.read_csv(path + 'df_test_collated.csv')

lm = LearningModel(df=df_train, target_variable='demand', split_ratio=0.2,
                    output_folder='../prob_results_ubr/',

                     # parameters for the phi.control
                     rel_method='range',
                     extr_type='high', coef=1.5,
                     relevance_pts=mat,

                     # parameters for the SmoteRegress, RandUnder, GaussNoiseRegress, DIBSRegress
                     rel=mat, thr_rel=0.8, Cperc="balance",
                     k=5, repl=False, dist='Manhattan', p=2, pert=0.1,

                     # the over/under sampling method
                     smogn=True, rand_under=False, smoter=False, gn=False, nosmote=False,

                     # scaling input and output
                     cols_drop=None, scale=True, scale_input=True, scale_output=False,
                     output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                     input_zscore=None, input_minmax=scaling[KEY], input_box=None, input_log=None,

                     testing_data=df_test,

                     grid=True, random_grid=False,
                     nb_folds_grid=3, nb_repeats_grid=None,
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

