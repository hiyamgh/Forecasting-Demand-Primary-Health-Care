from ngboost.distns import Exponential

possible_hyperparams_per_model = {

    'bootstrap': {
        'dropout': [0.01, 0.1, 0.3],
        'learning_rate': [0.01, 0.1, 0.3],
        'epochs': [100],
        'n_heads': [3, 5, 7]
    },
    'combined': {
        # 'dropout': [0.1, 0.3],
        # 'dropout': [0.0001, 0.001], this one
        'dropout': [0.001],
        'learning_rate': [0.1],
        # 'learning_rate': [0.001, 0.0001, 0.00001],
        'epochs': [100],
        'n_passes': [10, 100, 1000]

    },

    'mixture': {
        'dropout': [0.01, 0.1, 0.3],
        'learning_rate': [0.01, 0.1, 0.3],
        'epochs': [100],
        'n_mixtures': [3, 5, 7, 10]
    },

    'ngboost': {
        # 'Dist': [Exponential],
        # 'n_estimators': [1500],
        # 'learning_rate': [0.1],
        # 'minibatch_frac': [1.0],
        # 'verbose': [False]
        'Dist': [Exponential],
        'n_estimators': [500, 1000, 1500, 2000],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1],
        'minibatch_frac': [1.0, 0.5],
        'verbose': [False]
    },

    'mc_dropout': {
        'n_hidden': [150],
        'n_epochs': [100],
        'num_hidden_layers': [4],
        'epochx': [4],
        'normalize': [True],
        'tau': [0.15],
        # 'dropout_rate': [0.01, 0.05, 0.1, 0.3],
        'dropout_rate': [0.3],
        'T': [100]
    },

    'deep_ensemble': {
        'batch_size': [456],
        'optimizer_name': ['adam', 'grad_desc', 'ada_grad', 'ada_delta'],
        'learning_rate': [0.001],
        # 'max_iter': [30],
        'max_iter': [100],
        # 'sizes': [[256, 512, 1024, 1500, 1, 1]]
        'sizes': [[256, 512, 1, 1], [156, 256, 1, 1]]
    }
    #     # the ones that had no nan in loss
    #     'learning_rate': [0.001],
    #     'batch_size': [456],
    #     'optimizer_name': ['grad_desc'],
    #     # 'max_iter': [30],
    #     'max_iter': [100],
    #     'sizes': [[256, 512, 1024, 1500, 1, 1]]
    #     # 'sizes': [[100, 150, 250, 500, 1, 1]]
    #     # 'sizes': [[50, 30, 10, 1, 1]]
    # }
}


# from ngboost.distns import Exponential
#
# #         y_eval, uncertainty = bootstrap_evaluation(x=X_train, y=y_train, x_eval=X_test, dropout=0.3,
# #                                                  learning_rate=1e-3, epochs=100, n_heads=n_heads)
# possible_hyperparams_per_model = {
#
#     'bootstrap':{
#         'dropout': [0.3],
#         'learning_rate': [1e-1],
#         'epochs': [100],
#         'n_heads': [3]
#     },
#     'combined': {
#         'dropout': [0.3],
#         'learning_rate': [1e-3],
#         'epochs': [100],
#         'n_passes': [100]
#         # 'n_passes': [3]
#
#         # 'dropout': [0.3],
#         # 'learning_rate': [1e-3],
#         # 'epochs': [100],
#         # 'n_passes': [100]
#     },
#
#     'mixture':{
#         # 'dropout': [0.1],
#         # 'dropout': [0.01],
#         # 'dropout': [0.001],
#         'dropout': [0.1],
#         # 'learning_rate': [1e-3],
#         'learning_rate': [1e-3],
#         'epochs': [100],
#         'n_mixtures': [1]
#     },
#
#     'ngboost': {
#         'Dist': [Exponential],
#         'n_estimators': [1500],
#         'learning_rate': [0.1],
#         'minibatch_frac': [1.0],
#         'verbose': [False]
#         # 'Dist': [Exponential],
#         # 'n_estimators': [500, 1000, 1500, 2000],
#         # 'learning_rate': [0.0001, 0.001, 0.01, 0.1],
#         # 'minibatch_frac': [1.0, 0.5],
#         # 'verbose': [False]
#     },
#
#     'mc_dropout': {
#         'n_hidden': [150],
#         'n_epochs': [100],
#         'num_hidden_layers': [4],
#         'epochx': [4],
#         'normalize': [True],
#         'tau': [0.15],
#         # 'dropout_rate': [0.01, 0.05, 0.1, 0.3],
#         'dropout_rate': [0.3],
#         'T': [100]
#
#
#         # 'n_hidden': [150],
#         # 'n_epochs': [100],
#         # 'num_hidden_layers': [5],
#         # 'epochx': [500],
#         # 'normalize': [True],
#         # 'tau': [0.15],
#         # 'dropout_rate': [0.01],
#         # 'T': [100, 500, 1000]
#         #
#
#         # 'n_hidden': [100, 150, 200],
#         # # 'n_epochs': [40],
#         # 'n_epochs': [10, 15, 20],
#         # 'num_hidden_layers': [4, 5, 6],
#         #
#         # # Multiplier for the number of epochs for training. - not a parameter in the model
#         # 'epochx': [500],
#         # 'normalize': [True],
#         # # 'tau': [0.15],
#         # 'tau': [0.1, 0.15, 0.2],
#         # # 'dropout_rate': [0.01],
#         # 'dropout_rate': [0.005, 0.01, 0.05, 0.1],
#         # 'T': [100, 1000, 1500]
#         # # 'T': [1000]
#
#     },
#
#     'deep_ensemble': {
#
#         # the ones that had no nan in loss
#         'learning_rate': [0.0001],
#         'batch_size': [456],
#         'optimizer_name': ['adam'],
#         # 'max_iter': [30],
#         'max_iter': [80],
#         # 'sizes': [[256, 512, 1024, 1500, 1, 1]]
#         'sizes': [[256, 512, 1024, 1, 1]]
#
#         # # winning
#         # 'learning_rate': [0.0001],
#         # 'batch_size': [556, 646],
#         # 'optimizer_name': ['adam'],
#         # 'max_iter': [100, 150, 200],
#         # 'sizes': [[256, 512, 1024, 1, 1]]
#
#         # 'learning_rate': [0.0001, 0.000001],
#         # 'batch_size': [456, 556, 656],
#         # 'optimizer_name': ['adam'],
#         # 'max_iter': [100, 50, 40],
#         # 'sizes': [[256, 512, 1024, 1, 1], [200, 100, 80, 40, 1, 1]]
#     }
# }
#
#
#
# # from ngboost.distns import Exponential
# #
# # possible_hyperparams_per_model = {
# #
# #     'ngboost': {
# #         # 'Dist': [Exponential],
# #         # 'n_estimators': [1500],
# #         # 'learning_rate': [0.1],
# #         # 'minibatch_frac': [1.0],
# #         # 'verbose': [False]
# #         'Dist': [Exponential],
# #         'n_estimators': [500, 1000, 1500, 2000],
# #         'learning_rate': [0.0001, 0.001, 0.01, 0.1],
# #         'minibatch_frac': [1.0, 0.5],
# #         'verbose': [False]
# #     },
# #
# #     'mc_dropout': {
# #         'n_hidden': [150],
# #         'n_epochs': [100],
# #         'num_hidden_layers': [5],
# #         'epochx': [500],
# #         'normalize': [True],
# #         'tau': [0.15],
# #         'dropout_rate': [0.01],
# #         'T': [1000]
# #         # 'n_hidden': [100, 150, 200],
# #         # # 'n_epochs': [40],
# #         # 'n_epochs': [10, 15, 20],
# #         # 'num_hidden_layers': [4, 5, 6],
# #         #
# #         # # Multiplier for the number of epochs for training. - not a parameter in the model
# #         # 'epochx': [500],
# #         # 'normalize': [True],
# #         # # 'tau': [0.15],
# #         # 'tau': [0.1, 0.15, 0.2],
# #         # # 'dropout_rate': [0.01],
# #         # 'dropout_rate': [0.005, 0.01, 0.05, 0.1],
# #         # 'T': [100, 1000, 1500]
# #         # # 'T': [1000]
# #
# #     },
# #
# #     'deep_ensemble': {
# #         'learning_rate': [0.0001, 0.000001],
# #         'batch_size': [456, 556, 656],
# #         'optimizer_name': ['adam'],
# #         'max_iter': [100, 50, 40],
# #         'sizes': [[256, 512, 1024, 1, 1], [200, 100, 80, 40, 1, 1]]
# #     }
# #
# # }
