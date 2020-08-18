import pandas as pd
from cross_validation_smogn import LearningModel
from models_hyperparams_grid import possible_hyperparams_per_model as hyperparameters, models_to_test
from container import *

KEY = 'all_columns_minus_weather_minus_vdc'
c = 'extreme'

print('\n=============================================================================================================')
print('===================================        %s         =================================================' % KEY)
print('=============================================================================================================\n')

path = datasets[KEY]
df_train = pd.read_csv(path + 'df_train_collated.csv')
df_test = pd.read_csv(path + 'df_test_collated.csv')

# combination1 contains the combinations for running experiments with relevance matrix
# t is a tuple of the following structure: (threshold - distance - method - rel_mat_name)
for t in combination2:
    print('************************************* {} *************************************'.format(t))
    vec = get_bool_method(t[2])
    lm = LearningModel(df=df_train, target_variable='demand', split_ratio=0.2,
                    output_folder='../output_ubr_shallow/%s_%s/%s_%s_%s_%s/' % (KEY, c, t[2], t[0], t[1], t[3]),

                       # parameters for the phi.control :) :)
                       rel_method='extremes' if len(t) == 3 else 'range',
                       extr_type='high', coef=1.5,
                       relevance_pts=None if len(t) == 3 else get_rel_mat(t[3]),

                       # parameters for the SmoteRegress, RandUnder, GaussNoiseRegress, DIBSRegress
                       rel="auto" if len(t) == 3 else get_rel_mat(t[3]), thr_rel=t[0], Cperc="extreme",
                     # Cperc='balance',
                     k=5, repl=False, dist=t[1], p=2, pert=0.1,

                     # the over/under sampling method
                     smogn=vec[0], rand_under=vec[1], smoter=vec[2], gn=vec[3], nosmote=vec[4],

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

    for model in models_to_test:
        model_name = models_to_test[model]
        print('\n********** Results for %s **********' % model_name)

        # apply cross validation for the current model
        lm.cross_validation(model, hyperparameters[model_name], model_name)

        lm.errors_to_csv()

        lm.rare_statistics_to_csv()

    print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

