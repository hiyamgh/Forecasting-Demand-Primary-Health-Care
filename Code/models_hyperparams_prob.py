from ngboost.distns import Exponential

possible_hyperparams_per_model = {

    'bootstrap':{
        'dropout': [0.3, 0.1, 0.01, 0.001, 0.0001],
        'learning_rate': [0.3, 0.1, 0.001, 0.0001, 0.00001],
        'epochs': [100],
        'n_heads': [3, 5, 7]
    },
    'combined': {
        'dropout': [0.001, 0.0001, 0.00001],
        'learning_rate': [0.001, 0.0001, 0.00001],
        'epochs': [100],
        'n_passes': [10, 100, 1000]

    },

    'mixture': {
        'dropout': [0.001, 0.0001, 0.00001],
        'learning_rate': [0.0001, 0.00001],
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

        # the ones that had no nan in loss
        'learning_rate': [0.0001],
        'batch_size': [456],
        'optimizer_name': ['adam'],
        # 'max_iter': [30],
        'max_iter': [100],
        # 'sizes': [[256, 512, 1024, 1500, 1, 1]]
        'sizes': [[256, 512, 1024, 1, 1]]
    }
}

