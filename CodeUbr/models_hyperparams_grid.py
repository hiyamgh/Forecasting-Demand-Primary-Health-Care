import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.svm import SVR
from sklearn.svm import LinearSVR, NuSVR
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, BaggingRegressor
from xgboost import XGBRegressor


# random search first to suggest a few hyperparameters followed by grid search to zoom in
#Which features to tune
#Which starting point and which ending point for the space of each parameters
#Which hop per space

possible_hyperparams_per_model = {
        'lasso': {
            'alpha': [1, 0.1, 0.2, 0.5, 0.7, 0.01, 0.001, 0.0001]
        },
        'decision_tree': {
            'max_depth': np.random.randint(low=1, high=25, size=5),
            'min_samples_split': np.random.uniform(low=0, high=1, size=5),
            'min_samples_leaf': np.random.randint(low=1, high=10, size=5),
            'max_features': np.array(range(1, 10, 2))
        },
        'random_forest': {
            'max_depth': [10, 15, 20],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_split': [2, 5, 10, 15, 100],
            'min_samples_leaf': [1, 2, 5, 10]
            # 'n_estimators': range(1, 200, 10),
            # 'max_depth': range(1, 200, 10),
            # 'min_samples_split': np.arange(0.00001, 1, 0.1),
            # 'min_samples_leaf': np.arange(0.00001, 1, 0.1),
            # 'max_features': ['auto', 'sqrt', 'log2', None],
            # 'max_leaf_nodes': [None, range(1, 60, 10)],
            # 'max_samples': np.arange(0.00001, 1, 0.1)
        },
        'ada_boost': {
            'n_estimators': [5, 10, 15],
            'learning_rate': [0.1, 0.3],
            'loss': ['linear', 'square', 'exponential']
        },
        'gradient_boost': {
            'n_estimators': [10, 15, 20, 25, 30],
            'max_features': [0.1, 0.25, 0.5, 0.75, 1],
            'max_depth': [8, 10],
            'learning_rate': [0.1],
            # 'loss': ['ls', 'lad', 'huber', 'quantile'],
            'subsample': [1]
        },
        'cat_boost': {
            'depth': [6, 8, 10],
            'learning_rate': [0.01, 0.05, 0.1],
            'iterations': [30, 50, 100],
            'verbose': [0]
        },
        'xg_boost': {
            'max_depth': [4],
            'min_child_weight':  [1, 3, 5],
            'gamma': [0.2, 0.3],
            'colsample_bytree': [0.3, 0.4],
            'objective': ['reg:squarederror', 'reg:squaredlogerror']
        },
        'svr': {'C': [0.1, 1, 10],
                'gamma': [0.1, 1, 10]
                },
        'ridge': {'alpha': [1, 0.1, 0.2, 0.5, 0.7, 0.1, 0.01, 0.001, 0.0001, 0]},
        'elastic_net': {
            'max_iter': [5, 10, 15],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
            'l1_ratio': np.arange(0.0, 1.0, 0.1)},
        'extra_trees': {
            "n_estimators": [80],
            'max_depth': [30,],
            'max_features': ['auto', 'sqrt', 'log2'],
            'min_samples_split': [0.01, 0.05, 0.10],
            'min_samples_leaf': [0.005, 0.05, 0.10],
        },
        'bagging': {
            "base_estimator": [DecisionTreeRegressor(max_depth=8)],
            "n_estimators": [200],
            "max_features": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
        'sgd': {
            "alpha": [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.25, 0.50, 0.75, 1.0],
            "penalty": ["l1", "l2"],
            "loss": ['squared_loss', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
        },
        'linear_svr': {
            "C": [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100],
            "loss": ['epsilon_insensitive', 'squared_epsilon_insensitive']
        },
        'nu_svr': {
            "nu": [0.25, 0.50, 0.75],
            "kernel": ["linear", "rbf", "poly"],
            "degree": [1, 2, 3, 4, 5, 6],
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
            # RandomForestRegressor: 'random_forest',
            XGBRegressor: 'xg_boost',
            # CatBoostRegressor: 'cat_boost',
            ExtraTreesRegressor: 'extra_trees',
            BaggingRegressor: 'bagging',
            SGDRegressor: 'sgd',
            LinearSVR: 'linear_svr',
            # # NuSVR: 'nu_svr',
        }

