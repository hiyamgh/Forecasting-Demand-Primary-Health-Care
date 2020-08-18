import pandas as pd
from cross_validation_smogn_lime import LearningModel
from models_hyperparams_grid import possible_hyperparams_per_model as hyperparameters, models_to_test
from container import *

KEY = 'all_columns_minus_weather_minus_lags_minus_civilians'
c = 'balance'

path = datasets[KEY]
df_train = pd.read_csv(path + 'df_train_collated.csv')
df_test = pd.read_csv(path + 'df_test_collated.csv')

meth = 'smogn'
threshold = 0.95
distance = 'Manhattan'
relmat = 'rel_mat1'
model = 'bagging'

lm = LearningModel(df=df_train, target_variable='demand', split_ratio=0.2,
                    output_folder='../output_additional_unscaled/%s_%s/%s_%s_%s_%s/' % (KEY, c, meth, str(threshold), distance, relmat),

                     rel_method='range',
                     extr_type='high', coef=1.5,
                     relevance_pts=get_rel_mat(relmat),

                     # parameters for the SmoteRegress, RandUnder, GaussNoiseRegress, DIBSRegress
                     rel=get_rel_mat(relmat), thr_rel=threshold, Cperc=c,
                     k=5, repl=False, dist=distance, p=2, pert=0.1,

                     # the over/under sampling method
                     smogn=True, rand_under=False, smoter=False, gn=False, nosmote=False,

                     # scaling input and output
                     cols_drop=None, scale=True, scale_input=True, scale_output=False,
                     output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                     input_zscore=None, input_minmax=scaling[KEY], input_box=None, input_log=None,

                     service_name=None, mohafaza=None,

                     testing_data=df_test,

                     grid=True, random_grid=False,
                     nb_folds_grid=10, nb_repeats_grid=10,
                     nb_folds_random=10, nb_repeats_random=5, nb_iterations_random=10,
                     save_errors_xlsx=True, save_validation=False)

for m in models_to_test:
    model_name = models_to_test[m]

    if model_name == model:
        print('\n********** Results for %s **********' % model_name)

        # apply cross validation for the current model
        lm.cross_validation(m, hyperparameters[model_name], model_name)

        lm.errors_to_csv()

        lm.rare_statistics_to_csv()

        break

print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')