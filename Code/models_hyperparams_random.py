import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR, NuSVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor


# random search first to suggest a few hyperparameters followed by grid search to zoom in
#Which features to tune
#Which starting point and which ending point for the space of each parameters
#Which hop per space

possible_hyperparams_per_model = {
        'lasso': {
            'clf__alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        'decision_tree': {
            'clf__max_depth': range(5, 200, 10),
            'clf__min_samples_split': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__min_samples_leaf': [0.2, 0.3, 0.4, 0.5, 1],
            'clf__max_features': ['auto', 'sqrt', 'log2', None],
            'clf__max_leaf_nodes': [None, 10, 20, 30, 40, 50, 60]
        },
        'random_forest': {
            'clf__n_estimators': range(5, 200, 10),
            'clf__max_depth': range(5, 200, 10),
            'clf__min_samples_split': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__min_samples_leaf': [0.2, 0.3, 0.4, 0.5, 1],
            'clf__max_features': ['auto', 'sqrt', 'log2', None],
            'clf__max_leaf_nodes': [None, 10, 20, 30, 40, 50, 60]
        },
        'ada_boost': {
            'clf__base_estimator': [DecisionTreeRegressor(max_depth=ii) for ii in range(10, 110, 10)],
            'clf__n_estimators': range(50, 200, 10),
            'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__loss': ['linear', 'square', 'exponential'],
        },
        'gradient_boost': {
            'clf__loss': ['ls', 'lad', 'huber', 'quantile'],
            'clf__learning_rate': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__n_estimators': range(100, 350, 10),
            'clf__max_depth': range(5, 200, 10),
            'clf__min_samples_split': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__min_samples_leaf': [0.2, 0.3, 0.4, 0.5, 1],
            'clf__max_features': ['auto', 'sqrt', 'log2', None],
            'clf__max_leaf_nodes': [None, 10, 20, 30, 40, 50, 60]
        },
        'cat_boost': {
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100],
            'verbose': [0]
        },
        'xg_boost': {
            'clf__eta': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__gamma': range(10, 500, 10),
            'clf__max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
            'clf__min_child_weight':  range(10, 500, 10),
            # 'clf__colsample_bytree': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            # 'clf__colsample_bylevel': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            # 'clf__colsample_bynode': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__colsample_by*': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__alpha': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__objective': ['reg:squarederror']
        },
        'svr': {
            'clf__kernel': ['rbf', 'linear', 'poly'],
            'clf__gamma': ['scale', 'auto'],
            'clf__C': [0.0001, 0.001, 0.01, 1, 10, 100, 1000],
            'clf__shrinking': [True, False],
        },
        'ridge': {
            'clf__alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        },
        'elastic_net': {
            'clf__alpha': [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__l1_ratio': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__fit_intercept': [True, False],
            'clf__normalize': [True, False],
        },

        'extra_trees': {
            'clf__n_estimators': range(10, 200, 10),
            'clf__max_depth': range(5, 200, 10),
            'clf__max_features': ['auto', 'sqrt', 'log2', None],
            'clf__bootstrap': [True, False]
        },
        'bagging': {
            'clf__base_estimator': [DecisionTreeRegressor(max_depth=ii) for ii in range(10, 110, 10)],
            'clf__n_estimators': range(10, 200, 10),
            'clf__max_features': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__bootstrap': [True, False],
            'clf__bootstrap_features': [True, False],
        },
        'sgd': {
            'clf__loss': ['squared_loss', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
            'clf__penalty': ['l2', 'l1', 'none', 'elasticnet'],
            'clf__alpha': [0.00001, 0.0001, 0.001, 0.1, 0.2, 0.3, 0.4, 0.50, 0.6, 0.7, 0.8, 0.9, 1],
            'clf__l1_ratio': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__fit_intercept': [True, False],
        },
        'linear_svr': {
            "clf__C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            'clf__loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
        },
        'nu_svr': {
            'clf__nu': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            'clf__C': [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            'clf__kernel': ['rbf', 'linear', 'poly'],
            'clf__degree': [2, 3, 4],
            'clf__gamma': ['scale', 'auto'],
        }
    }

# the set of models we will use. This dictionary maps each sklearn model to a string
# designating its name. The string will be used to name the plots by the models used
# to make up these plots
models_to_test = {
            Lasso: 'lasso',
            Ridge: 'ridge',
            ElasticNet: 'elastic_net',
            # SVR: 'svr',
            AdaBoostRegressor: 'ada_boost',
            GradientBoostingRegressor: 'gradient_boost',
            DecisionTreeRegressor: 'decision_tree',
            RandomForestRegressor: 'random_forest',
            XGBRegressor: 'xg_boost',
            # CatBoostRegressor: 'cat_boost',
            ExtraTreesRegressor: 'extra_trees',
            BaggingRegressor: 'bagging',
            SGDRegressor: 'sgd',
            LinearSVR: 'linear_svr',
            NuSVR: 'nu_svr',
        }

