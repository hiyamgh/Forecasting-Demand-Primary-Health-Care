import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")
import itertools as it
from sklearn.model_selection import KFold, RepeatedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import pickle
from scipy.spatial import distance
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from scipy.stats.stats import pearsonr, spearmanr
import pandas as pd
import scipy.special
import scipy.stats


class LearningModel:
    def __init__(self, df, target_variable, split_ratio: float,
                    output_folder,
                    cols_drop=None, scale=True, scale_input=True, scale_output=False,
                    output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                    input_zscore=None, input_minmax=None, input_box=None, input_log=None,
                    service_name=None, mohafaza=None, testing_data=None,
                    grid=True, random_grid=False,
                    nb_folds_grid=None, nb_repeats_grid=None, nb_folds_random=None,
                    nb_repeats_random=None, nb_iterations_random=None,
                    loo=False,
                    save_errors_xlsx=True,
                    save_validation=False):

        # data frames. If df_test is None, df will be split into training and testing according to split_ratio
        # Otherwise, df will be training, df_test will be testing
        self.df = df
        self.testing_data = testing_data

        # drop NaN values
        self.df = self.df.dropna()
        if self.testing_data is not None:
            self.testing_data = self.testing_data.dropna()

        if self.testing_data is None:
            nb_rows_test = int(round(len(self.df) * split_ratio))
            nb_rows_train = len(self.df) - nb_rows_test

            self.df_train = self.df[0: nb_rows_train]
            self.df_test = self.df[nb_rows_train:]
            print('first %d rows for training, last %d rows for testing' % (nb_rows_train, nb_rows_test))
        else:
            self.df_train = self.df
            self.df_test = self.testing_data
            print('param df is the training, param df_test is the testing ...')

        # original testing data (the testing data before dropping columns from it)
        # needed when attaching the 'predicted' column
        self.df_test_orig = self.df_test

        # output folder to save plots and data
        self.output_folder = output_folder

        # save the training and testing datasets before doing anything
        self.save_train_test_before_modeling()

        self.target_variable = target_variable

        # list of columns to drop
        self.cols_drop = cols_drop
        if self.cols_drop is not None:
            self.df_train = self.df_train.drop(self.cols_drop, axis=1)
            self.df_test = self.df_test.drop(self.cols_drop, axis=1)
            print('list of columns used in modeling')
            print(list(self.df_test.columns.values))

        print('shuffling the 80% training before cv ...')
        self.df_train = self.df_train.sample(frac=1, random_state=42)

        # output folder
        self.output_folder = output_folder

        # scaling input & output
        self.scale = scale
        self.scale_input = scale_input
        self.scale_output = scale_output

        # specify scaling method for output
        self.output_zscore = output_zscore
        self.output_minmax = output_minmax
        self.output_box = output_box
        self.output_log = output_log

        # specify scaling method for input
        self.input_zscore = input_zscore
        self.input_minmax = input_minmax
        self.input_box = input_box
        self.input_log = input_log

        # related to cross validation
        self.grid = grid
        self.random_grid = random_grid
        self.nb_folds_grid = nb_folds_grid
        self.nb_repeats_grid = nb_repeats_grid
        self.nb_folds_random = nb_folds_random
        self.nb_repeats_random = nb_repeats_random
        self.nb_iterations_random = nb_iterations_random
        self.loo = loo
        self.split_ratio = split_ratio

        # save error metrics to xlsx sheet
        self.save_errors_xlsx = save_errors_xlsx
        self.save_validation = save_validation

        if self.save_errors_xlsx:
            self.results = pd.DataFrame(columns=['r2', 'adj-r2', 'rmse', 'mse', 'mae', 'mape',
                                                     'avg_%s' % self.target_variable,
                                                     'pearson', 'spearman', 'distance', 'coefficients'])
        else:
            self.results = None

        if self.save_validation:
            self.results_validation = pd.DataFrame(columns=['r2', 'adj-r2',
                                                            'rmse', 'mse', 'mae', 'mape'])
        else:
            self.results_validation = None

        # if grid_search=True and random_search_then_grid=True
        if self.grid and self.random_grid:
            raise ValueError('you cannot set both `grid` and `random_grid` to True. Either one must be False')

        # if grid=False and random_search_then_grid_search=False
        elif not self.grid and not self.random_grid:
            raise ValueError('you cannot set both `grid` and `random_grid` to False. Either one must be True')

        elif self.grid and not self.random_grid:
            if self.nb_folds_grid is None:
                raise ValueError('Please set nb_folds_grid to a number')
        else:
            if self.nb_iterations_random is None or self.nb_folds_random is None:
                raise ValueError('Please specify\n1.nb_iterations_random\n'
                                 '2.nb_folds_random\n3.nb_repeats_random(if needed)')

        # service_name & mohafaza for MoPH
        self.service_name = service_name
        self.mohafaza = mohafaza

        df_without_target = self.df_train
        df_without_target = df_without_target.drop([self.target_variable], axis=1)
        self.feature_names = list(df_without_target.columns.values)
        print(self.feature_names)

        # numpy arrays X_train, y_train, X_test, y_test
        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.target_variable])

        # get the list of indices of columns for each scaling type
        self.idx_zscore, self.idx_minmax, self.idx_box, self.idx_log = None, None, None, None

        if self.input_zscore is not None:
            self.idx_zscore = list(range(self.input_zscore[0], self.input_zscore[1]))

        if self.input_minmax is not None:
            self.idx_minmax = list(range(self.input_minmax[0], self.input_minmax[1]))

        if self.input_box is not None:
            self.idx_box = list(range(self.input_box[0], self.input_box[1]))

        if self.input_log is not None:
            self.idx_log = list(range(self.input_log[0], self.input_log[1]))

        cols_coefs = self.feature_names + ['y_intercept']
        self.coefficients = pd.DataFrame(columns=cols_coefs)

    def save_train_test_before_modeling(self):
        ''' save the training and testing data frames before any processing happens to them '''
        path = self.output_folder + 'train_test_before_modeling/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.df_train.to_csv(path + 'training.csv', index=False)
        self.df_test.to_csv(path + 'testing.csv', index=False)

    def cross_validation(self, model_used, hyperparams, model_name):

        # if grid=True and random_grid=False
        if self.grid and not self.random_grid:
            self.cross_validation_grid(model_used, hyperparams, model_name)

        # if grid=False and random_grid=True
        else:
            self.cross_validation_random_grid(model_used, hyperparams, model_name)

    def inverse_boxcox(self, y_box, lambda_):
        pred_y = np.power((y_box * lambda_) + 1, 1 / lambda_) - 1
        return pred_y

    def cross_validation_random_grid(self, model_used, hyperparams, model_name):

        print('\n********** Random Search Results for %s **********' % model_name)
        if type(self.target_variable) is str:
            X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

            # initialize pipe
            pipe = None

            # will assume that user either want to do z-score alone, minmax alone, both, or not at all
            # (DISCARD OPTION FOR BOX AND LOG)
            if self.scale:
                if self.scale_input:
                    if self.input_zscore is not None and self.input_minmax is not None:
                        # print('1st condition')
                        ct = ColumnTransformer([('standard', StandardScaler(), self.idx_zscore),
                                                ('minmax', MinMaxScaler(), self.idx_minmax)], remainder='passthrough')
                        pipe = Pipeline(steps=[('preprocessor', ct), ('clf', model_used())])

                    elif self.input_zscore is not None and self.input_minmax is None:
                        # print('2nd condition')
                        ct = ColumnTransformer([('standard', StandardScaler(), self.idx_zscore)],
                                               remainder='passthrough')
                        pipe = Pipeline(steps=[('preprocessor', ct), ('clf', model_used())])

                    elif self.input_zscore is None and self.input_minmax is not None:
                        # print('3rd condition')
                        ct = ColumnTransformer([('minmax', MinMaxScaler(), self.idx_minmax)], remainder='passthrough')
                        pipe = Pipeline(steps=[('preprocessor', ct), ('clf', model_used())])

                    else:
                        # print('4th condition')
                        pipe = model_used()

            else:
                # print('4th condition')
                pipe = model_used()

            # if nb_repeats is None, KFold cv will be done
            if self.nb_repeats_random is None:
                cv_splitter = KFold(n_splits=self.nb_folds_random)
                print('Running Random Search using %d-folds with %d iterations' % (self.nb_folds_random, self.nb_iterations_random))

            # if nb_repeats is not None, repeated KFold will be done
            else:
                cv_splitter = RepeatedKFold(n_splits=self.nb_folds_random, n_repeats=self.nb_repeats_random)
                print('Running Random Search using %d-folds-%d-repeats with %d iterations' % (self.nb_folds_random, self.nb_repeats_random, self.nb_iterations_random))

            randomized_search = RandomizedSearchCV(pipe, hyperparams, random_state=1, n_iter=self.nb_iterations_random, cv=cv_splitter)
            randomized_search.fit(X_train, y_train)

            print('\nparameters chosen by random search for %s:\n' % model_name)
            print(randomized_search.best_params_)

            grid = randomized_search.best_params_
            # replace __clf with '' for grid search
            grid_new = {}
            for param in grid:
                param_new = param.replace('clf__', '')
                grid_new[param_new] = grid[param]

            grid_new = self.random_to_grid(model_name, grid_new)

            print('\nparameters that will be used for Grid Search: \n')
            print(grid_new)

            self.cross_validation_grid(model_used, grid_new, model_name)

    def random_to_grid(self, model_name, random_params):
        grid_parameters = {}
        if model_name == 'lasso' or model_name == 'ridge':
            alpha = random_params['alpha']
            grid_parameters = {
                'alpha': [alpha - 0.1, alpha, alpha + 0.1] if alpha >= 0.2
                else [round(alpha - 0.09, 2), alpha, round(alpha + 0.9, 1)] if alpha == 0.1
                else [round(alpha - 0.009, 3), alpha, round(alpha + 0.09, 2)] if alpha == 0.01
                else [round(alpha - 0.0009, 4), alpha, round(alpha + 0.009, 3)]
            }

        if model_name == 'decision_tree':
            max_depth = random_params['max_depth']
            min_samples_split = random_params['min_samples_split']
            min_samples_leaf = random_params['min_samples_leaf']
            max_features = random_params['max_features']
            max_leaf_nodes = random_params['max_leaf_nodes']

            grid_parameters = {
                'max_depth': [max_depth - 3, max_depth, max_depth + 3],
                'min_samples_split': [round(min_samples_split - 0.1, 1), min_samples_split, round(min_samples_split + 0.1, 1)] if min_samples_split >= 0.2
                else [round(min_samples_split - 0.09, 2), min_samples_split, round(min_samples_split + 0.9, 1)] if min_samples_split == 0.1
                else [round(min_samples_split - 0.009, 3), min_samples_split, round(min_samples_split + 0.09, 2)] if min_samples_split == 0.01
                else [round(min_samples_split - 0.0009, 4), min_samples_split, round(min_samples_split + 0.009, 3)],

                'max_features': [max_features],
                'min_samples_leaf': [min_samples_leaf - 0.1, min_samples_leaf, 1] if min_samples_leaf != 1 else [min_samples_leaf],
                'max_leaf_nodes': [max_leaf_nodes - 5, max_leaf_nodes,
                                   max_leaf_nodes + 5] if max_leaf_nodes is not None else [max_leaf_nodes]
            }

        if model_name == 'random_forest':

            n_estimators = random_params['n_estimators']
            max_depth = random_params['max_depth']
            min_samples_split = random_params['min_samples_split']
            min_samples_leaf = random_params['min_samples_leaf']
            max_features = random_params['max_features']
            max_leaf_nodes = random_params['max_leaf_nodes']

            grid_parameters = {
                'n_estimators': [n_estimators - 5, n_estimators, n_estimators + 5],
                'max_depth': [max_depth - 3, max_depth, max_depth + 3],
                'min_samples_split': [round(min_samples_split - 0.1, 1), min_samples_split, round(min_samples_split + 0.1, 1)] if min_samples_split >= 0.2
                else [round(min_samples_split - 0.09, 2), min_samples_split, round(min_samples_split + 0.9, 1)] if min_samples_split == 0.1
                else [round(min_samples_split - 0.009, 3), min_samples_split, round(min_samples_split + 0.09, 2)] if min_samples_split == 0.01
                else [round(min_samples_split - 0.0009, 4), min_samples_split, round(min_samples_split + 0.009, 3)],
                # 'min_samples_split': [min_samples_split - 0.1, min_samples_split, min_samples_split + 0.1],
                'max_features': [max_features],
                'min_samples_leaf': [min_samples_leaf - 0.1, min_samples_leaf, 1] if min_samples_leaf != 1 else [min_samples_leaf],
                'max_leaf_nodes': [max_leaf_nodes - 5, max_leaf_nodes, max_leaf_nodes + 5] if max_leaf_nodes is not None else [max_leaf_nodes]
            }

        if model_name == 'ada_boost':
            base_estimator = random_params['base_estimator']
            n_estimators = random_params['n_estimators']
            learning_rate = random_params['learning_rate']
            loss = random_params['loss']

            chosen_max_depth = base_estimator.get_params()['max_depth']
            est1 = DecisionTreeRegressor(max_depth=chosen_max_depth)
            est2 = DecisionTreeRegressor(max_depth=chosen_max_depth - 5)
            est3 = DecisionTreeRegressor(max_depth=chosen_max_depth + 5)
            grid_parameters = {
                'base_estimator': [est1, est2, est3],
                'n_estimators': [n_estimators - 10, n_estimators, n_estimators + 10],
                'learning_rate': [round(learning_rate - 0.1, 1), learning_rate, round(learning_rate + 0.1, 1)] if learning_rate >= 0.2
                else [round(learning_rate - 0.09, 2), learning_rate, round(learning_rate + 0.9, 1)] if learning_rate == 0.1
                else [round(learning_rate - 0.009, 3), learning_rate, round(learning_rate + 0.09, 2)] if learning_rate == 0.01
                else [round(learning_rate - 0.0009, 4), learning_rate, round(learning_rate + 0.009, 3)],
                'loss': [loss]
            }

        if model_name == 'gradient_boost':
            loss = random_params['loss']
            learning_rate = random_params['learning_rate']
            n_estimators = random_params['n_estimators']
            min_samples_split = random_params['min_samples_split']
            min_samples_leaf = random_params['min_samples_leaf']
            max_depth = random_params['max_depth']
            max_features = random_params['max_features']
            max_leaf_nodes = random_params['max_leaf_nodes']

            grid_parameters = {
                'loss': [loss],
                'learning_rate': [round(learning_rate - 0.1, 1), learning_rate, round(learning_rate + 0.1, 1)] if learning_rate >= 0.2
                else [round(learning_rate - 0.09, 2), learning_rate, round(learning_rate + 0.9, 1)] if learning_rate == 0.1
                else [round(learning_rate - 0.009, 3), learning_rate, round(learning_rate + 0.09, 2)] if learning_rate == 0.01
                else [round(learning_rate - 0.0009, 4), learning_rate, round(learning_rate + 0.009, 3)],
                'n_estimators': [n_estimators - 5, n_estimators, n_estimators + 5],
                'max_depth': [max_depth - 3, max_depth, max_depth + 3],
                'min_samples_split': [round(min_samples_split - 0.1, 1), min_samples_split, round(min_samples_split + 0.1, 1)] if min_samples_split >= 0.2
                else [round(min_samples_split - 0.09, 2), min_samples_split, round(min_samples_split + 0.9, 1)] if min_samples_split == 0.1
                else [round(min_samples_split - 0.009, 3), min_samples_split, round(min_samples_split + 0.09, 2)] if min_samples_split == 0.01
                else [round(min_samples_split - 0.0009, 4), min_samples_split, round(min_samples_split + 0.009, 3)],
                'max_features': [max_features],
                'min_samples_leaf': [min_samples_leaf - 0.1, min_samples_leaf, 1] if min_samples_leaf != 1 else [min_samples_leaf],
                'max_leaf_nodes': [max_leaf_nodes - 5, max_leaf_nodes, max_leaf_nodes + 5] if max_leaf_nodes is not None else [max_leaf_nodes]
            }

        if model_name == 'xg_boost':
            eta = random_params['eta']
            gamma = random_params['gamma']
            max_depth = random_params['max_depth']
            min_child_weight = random_params['min_child_weight']
            colsample_by = random_params['colsample_by*']
            # colsample_bytree = random_params['colsample_bytree']
            # colsample_bylevel = random_params['colsample_bylevel']
            # colsample_bynode = random_params['colsample_bynode']
            alpha = random_params['alpha']
            objective = random_params['objective']

            grid_parameters = {
                'eta': [round(eta - 0.1, 1), eta, eta + 0.1] if eta >= 0.2
                else [round(eta - 0.09, 2), eta, round(eta + 0.9, 1)] if eta == 0.1
                else [round(eta - 0.009, 3), eta, round(eta + 0.09, 2)] if eta == 0.01
                else [round(eta - 0.0009, 4), eta, round(eta + 0.009, 3)],
                'gamma': [gamma - 5, gamma, gamma + 5],
                'max_depth': [max_depth - 1, max_depth, max_depth + 1],
                'min_child_weight': [min_child_weight - 5, min_child_weight, min_child_weight + 5],
                # 'col_sample_bytree': [round(colsample_bytree - 0.1, 1), colsample_bytree, colsample_bytree + 0.1],
                # 'col_sample_bylevel': [round(colsample_bylevel - 0.1, 1), colsample_bylevel, colsample_bylevel + 0.1],
                # 'col_sample_bynode': [round(colsample_bynode - 0.1, 1), colsample_bynode, colsample_bynode + 0.1],
                'col_sample_by*': [round(colsample_by - 0.1, 1), colsample_by, round(colsample_by + 0.1, 1)],
                'alpha': [round(alpha - 0.1, 1), alpha, alpha + 0.1],
                'objective': [objective]
            }
        if model_name == 'svr':
            kernel = random_params['kernel']
            gamma = random_params['gamma']
            C = random_params['C']
            shrinking = random_params['shrinking']

            grid_parameters = {
                'kernel': [kernel],
                'gamma': [gamma],
                'C': [round(C - 0.009, 3), C, round(C + 0.9, 2)] if C == 0.01
                else [round(C - 0.0009, 4), C, round(C + 0.009, 3)] if C == 0.001
                else [round(C - 0.00009, 5), C, round(C + 0.00009, 4)],
                'shrinking': [shrinking]
            }

        if model_name == 'elastic_net':
            alpha = random_params['alpha']
            l1_ratio = random_params['l1_ratio']
            fit_intercept = random_params['fit_intercept']
            normalize = random_params['normalize']

            grid_parameters = {
                'alpha': [round(alpha - 0.1, 1), alpha, alpha + 0.1] if alpha >= 0.2
                else [round(alpha - 0.09, 2), alpha, round(alpha + 0.9, 1)] if alpha == 0.1
                else [round(alpha - 0.009, 3), alpha, round(alpha + 0.09, 2)] if alpha == 0.01
                else [round(alpha - 0.0009, 4), alpha, round(alpha + 0.009, 3)],
                'l1_ratio': [l1_ratio - 0.1, l1_ratio, l1_ratio + 0.1],
                'fit_intercept': [fit_intercept],
                'normalize': [normalize],
            }

        if model_name == 'extra_trees':

            n_estimators = random_params['n_estimators']
            max_depth = random_params['max_depth']
            max_features = random_params['max_features']
            bootstrap = random_params['bootstrap']

            grid_parameters = {
                'n_estimators': [n_estimators - 5, n_estimators, n_estimators + 5],
                'max_depth': [max_depth - 3, max_depth, max_depth + 3],
                'max_features': [max_features],
                'bootstrap': [bootstrap],
            }

        if model_name == 'bagging':

            base_estimator = random_params['base_estimator']
            n_estimators = random_params['n_estimators']
            max_features = random_params['max_features']
            bootstrap = random_params['bootstrap']
            bootstrap_features = random_params['bootstrap_features']

            chosen_max_depth = base_estimator.get_params()['max_depth']
            est1 = DecisionTreeRegressor(max_depth=chosen_max_depth)
            est2 = DecisionTreeRegressor(max_depth=chosen_max_depth - 10)
            est3 = DecisionTreeRegressor(max_depth=chosen_max_depth + 10)
            grid_parameters = {
                'base_estimator': [est1, est2, est3],
                'n_estimators': [n_estimators - 5, n_estimators, n_estimators + 5],
                'max_features': [round(max_features - 0.1, 1), max_features, max_features + 0.1] if max_features >= 0.2
                else [round(max_features - 0.09, 2), max_features, max_features + 0.1],
                'bootstrap': [bootstrap],
                'bootstrap_features': [bootstrap_features],
            }

        if model_name == 'sgd':

            loss = random_params['loss']
            penalty = random_params['penalty']
            alpha = random_params['alpha']
            l1_ratio = random_params['l1_ratio']
            fit_intercept = random_params['fit_intercept']

            grid_parameters = {
                'loss': [loss],
                'penalty': [penalty],
                'alpha': [round(alpha - 0.1, 1), alpha, alpha + 0.1] if alpha >= 0.2
                else [round(alpha - 0.09, 2), alpha, round(alpha + 0.9, 1)] if alpha == 0.1
                else [round(alpha - 0.009, 3), alpha, round(alpha + 0.09, 2)] if alpha == 0.01
                else [round(alpha - 0.0000009, 4), alpha, round(alpha + 0.009, 3)],
                'l1_ratio': [l1_ratio - 0.1, l1_ratio, l1_ratio + 0.1],
                'fit_intercept': [fit_intercept]
            }

        if model_name == 'linear_svr':

            C = random_params['C']
            loss = random_params['loss']
            grid_parameters = {
                'C': [round(C - 0.09, 2), C, round(C + 0.9, 1)] if C == 0.1
                else [round(C - 0.009, 3), C, round(C + 0.9, 2)] if C == 0.01
                else [round(C - 0.0009, 4), C, round(C + 0.009, 3)] if C == 0.001
                else [round(C - 0.00009, 5), C, round(C + 0.00009, 4)],
                'loss': [loss]
            }
        if model_name == 'nu_svr':

            nu = random_params['nu']
            C = random_params['C']
            kernel = random_params['kernel']
            gamma = random_params['gamma']
            degree = random_params['degree']

            grid_parameters = {
                'nu': [round(nu - 0.1, 1), nu, nu + 0.1],
                'C': [round(C - 0.09, 2), C, round(C + 0.9, 1)] if C == 0.1
                else [round(C - 0.009, 3), C, round(C + 0.9, 2)] if C == 0.01
                else [round(C - 0.0009, 4), C, round(C + 0.009, 3)] if C == 0.001
                else [round(C - 0.00009, 5), C, round(C + 0.00009, 4)] if C == 0.0001
                else [C - 0.9, C, C + 10] if C == 1.0
                else [1, 10, 100],
                'kernel': [kernel],
                'degree': [degree - 1, degree, degree + 1],
                'gamma': [gamma]
            }

        return grid_parameters

    def cross_validation_grid(self, model_used, possible_hyperparams, model_name):

        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        # training and testing data if we have single target variable
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        tempModels = []

        # specify the type of cv (kfold vs. repeated kfold)
        if self.loo:
            kf = LeaveOneOut() # leave one out does not take number of splits, because every sample will be once
        #     a testing sample
        else:
            if self.nb_repeats_grid is None:
                print('running %d-fold cross validation' % self.nb_folds_grid)
                kf = KFold(n_splits=self.nb_folds_grid, random_state=2652124)
            else:
                print('running %d-fold-%d-repeats cross validation' % (self.nb_folds_grid, self.nb_repeats_grid))
                kf = RepeatedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid, random_state=2652124)

        t1 = time.time()

        parameters = possible_hyperparams

        if model_name == 'decision_tree':
            if X_train.shape[1] < 10:
                print('changing the max_features of decision tree from 10 to %d' % X_train.shape[1])
                parameters['max_features'] = np.array(range(1, X_train.shape[1], 2))

        # hyper parameters loop
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):

            model = model_used(**parameter)
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []

            for train_index, test_index in kf.split(X_train):
                X_train_inner, X_val = X_train[train_index], X_train[test_index]
                y_train_inner, y_val = y_train[train_index], y_train[test_index]

                if self.scale:
                    X_train_inner, X_val, y_train_inner, y_val, scaler_out_final = self.scale_cols(X_train_inner, X_val,
                                                                                                   y_train_inner, y_val)

                model.fit(X_train_inner, y_train_inner)
                y_pred = model.predict(X_val)

                # AFTER PREDICTION, reverse the scaled output (if self.scale_output is on). The idea is to reverse scaling JUST BEFORE printing out the error metrics
                if self.scale_output:
                    if self.output_zscore or self.output_minmax:
                        y_pred = scaler_out_final.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
                        y_val = scaler_out_final.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
                    elif self.output_log:
                        y_pred = np.exp(y_pred.reshape(-1, 1))
                        y_val = np.exp(y_val)
                    else:
                        y_pred = self.inverse_boxcox(y_pred, self.y_train_lambda_)
                        y_val = self.inverse_boxcox(y_val, self.y_test_lambda_)

                if len(y_val) >= 2:
                    r2, adj_r2, rmse, mse, mae, mape, _, _, _ = get_stats(y_val, y_pred, X_val.shape[1])
                else:
                    r2, adj_r2, rmse, mse, mae, mape = get_stats(y_val, y_pred, X_val.shape[1])

                if self.loo:
                    pass
                else:
                    r2_scores.append(r2)
                    adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)

            if self.loo:
                tempModels.append(
                    [parameter, '-', '-', np.mean(rmse_scores), np.mean(mse_scores),
                     np.mean(mae_scores), np.mean(mape_scores)])
            else:
                tempModels.append(
                    [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores), np.mean(mse_scores),
                     np.mean(mae_scores), np.mean(mape_scores)])

        tempModels = sorted(tempModels, key=lambda k: k[3])
        winning_hyperparameters = tempModels[0][0]

        print('winning hyper parameters: ', str(winning_hyperparameters))
        if self.loo:
            print('Best Validation Scores:\nR^2: -\nAdj R^2: -\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
                  (tempModels[0][3], tempModels[0][4], tempModels[0][5], tempModels[0][6]))
        else:
            print('Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
                  (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
                   tempModels[0][6]))

        if self.save_errors_xlsx:
            if self.save_validation:
                self.results_validation.loc['%s' % model_name] = pd.Series({'r2': tempModels[0][1],
                                                          'adj-r2': tempModels[0][2],
                                                          'rmse': tempModels[0][3],
                                                          'mse': tempModels[0][4],
                                                          'mae': tempModels[0][5],
                                                          'mape': tempModels[0][6]})

        model = model_used(**winning_hyperparameters)

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        model.fit(X_train, y_train)
        coefficients = None

        # save the model
        models_folder = self.output_folder + 'trained_models/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        pkl_filename = "%s.pkl" % model_name
        with open(models_folder + pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        print('saved model to {} as {}.pkl'.format(models_folder, model_name))

        # get the coeficients and y-intercept
        if model_name in ['lasso', 'ridge', 'elastic_net', 'linear_svr']:
            coef_ = model.coef_
            intercept_ = model.intercept_

            regression_equation = ''
            count = 0
            for coef, feat in zip(list(coef_), self.feature_names):
                regression_equation += '%s*%f + ' % ('X%s' % str(count), coef)
                count += 1
            regression_equation += str(intercept_)
            coefficients = regression_equation

        y_pred = model.predict(X_test)

        if self.scale_output:
            if self.output_log:
                y_pred_reverse = np.exp(y_pred.reshape(-1, 1))
                y_test = np.exp(y_test)
            elif self.output_box:
                y_pred_reverse = self.inverse_boxcox(y_pred, self.y_train_lambda_)
                y_test = self.inverse_boxcox(y_test, self.y_test_lambda_)
            else:
                y_pred_reverse = scaler_out_final.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
                y_test = scaler_out_final.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

        t2 = time.time()
        time_taken_min = float(t2 - t1) / 60
        time_taken_sec = float(t2 - t1)

        if self.scale_output:
            output_dataset = self.create_output_dataset(y_pred_reverse, model_name,
                                                        self.output_folder + 'output_vector_datasets/')
        else:
            output_dataset = self.create_output_dataset(y_pred, model_name, self.output_folder + 'output_vector_datasets/')

        self.plot_actual_vs_predicted(output_dataset, model_name, self.output_folder + 'train_test_forecasts_lineplot/', 'predicted')
        self.plot_actual_vs_predicted_scatter_bisector(output_dataset, model_name, self.output_folder + 'train_test_forecasts_scatterplot_bisector/', 'predicted')
        self.produce_learning_curve(model_used, model_name, 10, self.output_folder + '/learning_curves/', parameters=winning_hyperparameters, nb_repeats=10)

        pearson, spearman, distance = '', '', ''
        if len(y_test) >= 2:
            r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = get_stats(y_test, y_pred, X_test.shape[1])
        else:
            r2, adj_r2, rmse, mse, mae, mape = get_stats(y_test, y_pred, X_test.shape[1])
        print('Testing Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape))

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))

        if pearson != '':
            print('Pearson Correlation: %.5f' % pearson)
            print('Spearman Correlation: %.5f' % spearman)
            print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        if self.save_errors_xlsx:
            row_name = model_name

            if coefficients is not None:
                self.results.loc[row_name] = pd.Series({'r2': r2, 'adj-r2': adj_r2, 'rmse': rmse, 'mse': mse,
                                                        'mae': mae, 'mape': mape,
                                                        'avg_%s' % self.target_variable: avg_target,
                                                        'pearson': pearson, 'spearman': spearman,
                                                        'distance': distance,
                                                        'coefficients': coefficients})
            else:
                self.results.loc[row_name] = pd.Series({'r2': r2, 'adj-r2': adj_r2, 'rmse': rmse, 'mse': mse,
                                                        'mae': mae, 'mape': mape,
                                                        'avg_%s' % self.target_variable: avg_target,
                                                        'pearson': pearson, 'spearman': spearman,
                                                        'distance': distance})

        if not os.path.exists(self.output_folder + '/winning_hyperparams/'):
            os.makedirs(self.output_folder + '/winning_hyperparams/')
        with open(self.output_folder + 'winning_hyperparams/%s_hyperparams.pickle' % model_name, 'wb') as handle:
            pickle.dump(winning_hyperparameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_output_dataset(self, y_pred, model_name, output_folder):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # JUST TO AVOID THE CURRENT BUG
        df_test_curr = self.df_test_orig
        if 'predicted' in list(df_test_curr.columns.values):
            df_test_curr = df_test_curr.drop('predicted', axis=1)

        # add the predicted value to the df
        target_loc = df_test_curr.columns.get_loc(self.target_variable)
        df_test_curr.insert(target_loc + 1, 'predicted', list(y_pred))

        if self.service_name is None and self.mohafaza is None:
            # df_train.to_csv('train_df.csv')
            df_test_curr.to_csv(output_folder + 'test_df_%s.csv' % model_name, index=False)
        else:
            # df_train.to_csv('%s_%s_train.csv' % (service_name, mohafaza))
            if not os.path.exists(output_folder + '/%s_%s/' % (self.service_name, self.mohafaza)):
                os.makedirs(output_folder + '/%s_%s/' % (self.service_name, self.mohafaza))
            df_test_curr.to_csv(output_folder + '%s_%s/test_%s.csv' % (self.service_name, self.mohafaza, model_name))

        return df_test_curr

    def scale_cols(self, X_train, X_test, y_train, y_test):

        # z-score scaling
        if self.scale_input:
            if self.input_zscore is not None:
                # apply Standard scaling to the specified columns.
                scaler = StandardScaler()
                X_train = X_train.astype('float64')
                X_test = X_test.astype('float64')

                X_train_zscaled = scaler.fit_transform(X_train[:, self.idx_zscore])
                X_test_zscaled = scaler.transform(X_test[:, self.idx_zscore])

                for i in range(len(self.idx_zscore)):
                    X_train[:, self.idx_zscore[i]] = X_train_zscaled[:, i]
                    X_test[:, self.idx_zscore[i]] = X_test_zscaled[:, i]

            if self.input_minmax is not None:
                # apply MinMax scaling to the specified columns.
                scaler = MinMaxScaler()
                if X_train.dtype != 'float64':
                    X_train = X_train.astype('float64')
                    X_test = X_test.astype('float64')

                X_train_minmaxscaled = scaler.fit_transform(X_train[:, self.idx_minmax])
                X_test_minmaxscaled = scaler.transform(X_test[:, self.idx_minmax])

                for i in range(len(self.idx_minmax)):
                    X_train[:, self.idx_minmax[i]] = X_train_minmaxscaled[:, i]
                    X_test[:, self.idx_minmax[i]] = X_test_minmaxscaled[:, i]

            if self.input_box is not None:
                # apply BoxCox transform to the specified columns.
                if X_train.dtype != 'float64':
                    X_train = X_train.astype('float64')
                    X_test = X_test.astype('float64')

                X_train_boxscaled = np.array([list(scipy.stats.boxcox(X_train[:, self.idx_box[i]])[0]) for i in range(len(self.idx_box))]).T
                X_test_boxscaled = np.array([list(scipy.stats.boxcox(X_test[:, self.idx_box[i]])[0]) for i in range(len(self.idx_box))]).T

                for i in range(len(self.idx_box)):
                    X_train[:, self.idx_box[i]] = X_train_boxscaled[:, i]
                    X_test[:, self.idx_box[i]] = X_test_boxscaled[:, i]

            if self.input_log is not None:
                # apply Log transform to the specified columns.

                if X_train.dtype != 'float64':
                    X_train = X_train.astype('float64')
                    X_test = X_test.astype('float64')

                X_train_logscaled = np.log(X_train[:, self.idx_log])
                X_test_logscaled = np.log(X_test[:, self.idx_log])

                for i in range(len(self.idx_log)):
                    X_train[:, self.idx_log[i]] = X_train_logscaled[:, i]
                    X_test[:, self.idx_log[i]] = X_test_logscaled[:, i]

        scaler_out_final = None

        if self.scale_output:
            if self.output_zscore:
                scaler_out = StandardScaler()
                y_train = y_train.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)

                y_train = scaler_out.fit_transform(y_train)
                y_test = scaler_out.transform(y_test)

                y_train = y_train.reshape(-1)
                y_test = y_test.reshape(-1)

                scaler_out_final = scaler_out

            elif self.output_minmax:
                scaler_out = MinMaxScaler()
                y_train = y_train.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)

                y_train = scaler_out.fit_transform(y_train)
                y_test = scaler_out.transform(y_test)

                y_train = y_train.reshape(-1)
                y_test = y_test.reshape(-1)

                scaler_out_final = scaler_out

            elif self.output_box:
                # first I get the best lambda from scipy.stats, use it scipy.special to use it in scipy.spcial.inverse_box
                y_train, self.y_train_lambda_ = scipy.stats.boxcox(y_train)
                y_test, self.y_test_lambda_ = scipy.stats.boxcox(y_test)

            else:
                if self.output_log:
                    y_train = np.log(y_train)
                    y_test = np.log(y_test)

        return X_train, X_test, y_train, y_test, scaler_out_final

    def produce_learning_curve(self, model, model_name, nb_splits, output_folder, parameters, nb_repeats=None):

        '''
        produce learning curve of a certain model, using either KFold or repeated KFold cross validation
        :param model: the model
        :param model_name: name of the model, string.
        :param nb_splits: number of splits in KFold
        :param output_folder: path to output folder. If doesn't exist, will be created at runtime
        :param nb_repeats: number of repeats in case of RepeatedKFold. By defualt None. If None,
        KFold will be used instead
        :return: saves the learning curve
        '''

        X_train, y_train = self.X_train, self.y_train
        pipe = None

        if self.scale:
            if self.scale_output:
                if self.output_zscore:
                    scaler = StandardScaler()
                    y_train = scaler.fit_transform(y_train)
                elif self.output_minmax:
                    scaler = MinMaxScaler()
                    y_train = scaler.fit_transform(y_train)
                elif self.output_log:
                    y_train = np.log(y_train)
                else:
                    y_train, _ = scipy.stats.boxcox(y_train)

            if self.scale_input:
                if self.input_zscore is not None and self.input_minmax is not None:
                    # print('1st condition')
                    ct = ColumnTransformer([('standard', StandardScaler(), self.idx_zscore),
                                            ('minmax', MinMaxScaler(), self.idx_minmax)], remainder='passthrough')
                    pipe = Pipeline(steps=[('preprocessor', ct), ('model', model(**parameters))])

                elif self.input_zscore is not None and self.input_minmax is None:
                    # print('2nd condition')
                    ct = ColumnTransformer([('standard', StandardScaler(), self.idx_zscore)], remainder='passthrough')
                    pipe = Pipeline(steps=[('preprocessor', ct), ('model', model(**parameters))])

                elif self.input_zscore is None and self.input_minmax is not None:
                    # print('3rd condition')
                    ct = ColumnTransformer([('minmax', MinMaxScaler(), self.idx_minmax)], remainder='passthrough')
                    pipe = Pipeline(steps=[('preprocessor', ct), ('model', model(**parameters))])

                else:
                    # print('4th condition')
                    pipe = model(**parameters)

        else:
            # print('4th condition')
            pipe = model(**parameters)

        if nb_repeats is None:
            cv = KFold(n_splits=nb_splits, random_state=2652124)
        else:
            cv = RepeatedKFold(n_splits=nb_splits, n_repeats=nb_repeats, random_state=2652124)

        # if box or log transform is needed, this must not necessarily be done in a pipeline manner
        # because same transformation is done for training and validation, UNLIKE z-score and minmax
        # whereby scaling must be done on training THEN taking the parameters and apply them
        # to the validation

        if self.scale:
            if self.scale_input:
                if self.input_box is not None:
                    # apply BoxCox transform to the specified columns.
                    if X_train.dtype != 'float64':
                        X_train = X_train.astype('float64')

                    X_train_boxscaled = np.array([list(scipy.stats.boxcox(X_train[:, self.idx_box[i]])[0]) for i in range(len(self.idx_box))]).T

                    for i in range(len(self.idx_box)):
                        X_train[:, self.idx_box[i]] = X_train_boxscaled[:, i]

                if self.input_log is not None:
                    # apply Log transform to the specified columns.
                    if X_train.dtype != 'float64':
                        X_train = X_train.astype('float64')

                    X_train_logscaled = np.log(X_train[:, self.idx_log])

                    for i in range(len(self.idx_log)):
                        X_train[:, self.idx_log[i]] = X_train_logscaled[:, i]

        train_sizes, train_scores, test_scores = learning_curve(pipe, X_train, y_train, cv=cv, scoring='neg_mean_squared_error')  # calculate learning curve values

        train_scores_mean = -train_scores.mean(axis=1)
        test_scores_mean = -test_scores.mean(axis=1)

        plt.figure()
        plt.xlabel("Number of Training Samples")
        plt.ylabel("MSE")

        plt.plot(train_sizes, train_scores_mean, label="training")
        plt.plot(train_sizes, test_scores_mean, label="validation")
        plt.legend()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + '%s_learning_curve.png' % model_name)
        plt.close()

    def plot_actual_vs_predicted(self, df, model_name, output_folder, predicted_variable):

        plt.plot(list(range(1, len(df) + 1)), df[self.target_variable], color='b', label='actual')
        plt.plot(list(range(1, len(df) + 1)), df[predicted_variable], color='r', label='predicted')
        plt.legend(loc='best')
        # plt.suptitle('actual vs. predicted forecasts for %s in %s' % (self.service_name, self.mohafaza))

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + 'forecasts_%s' % model_name)
        plt.close()

    def plot_actual_vs_predicted_scatter_bisector(self, df, model_name, output_folder, predicted_variable,):
        fig, ax = plt.subplots()

        ax.scatter(df[self.target_variable], df[predicted_variable], c='black')

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        # plt.suptitle('actual vs. predicted forecasts')

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        plt.savefig(output_folder + 'scatter_%s' % model_name)
        plt.close()

    def errors_to_csv(self):
        ''' saves the error metrics (stored in `results`) as a csv file '''
        if self.results is not None:
            errors_df = self.results
            errors_df = errors_df.sort_values(by=['rmse'])
            path = self.output_folder + 'error_metrics_csv/'
            if not os.path.exists(path):
                os.makedirs(path)
            errors_df.to_csv(path + 'errors.csv')
            if self.results_validation is not None:
                validation_errors_df = self.results_validation
                validation_errors_df = validation_errors_df.sort_values(by=['rmse'])
                validation_errors_df.to_csv(path + 'errors_validation.csv')


def create_output_dataset(df_test_curr, y_pred, model_name, output_folder, target_variable, service_name=None, mohafaza=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # add the predicted value to the df
    target_loc = df_test_curr.columns.get_loc(target_variable)
    df_test_curr.insert(target_loc + 1, 'predicted', list(y_pred))

    if service_name is None and mohafaza is None:
        # df_train.to_csv('train_df.csv')
        df_test_curr.to_csv(output_folder + 'test_df_%s.csv' % model_name, index=False)
    else:
        # df_train.to_csv('%s_%s_train.csv' % (service_name, mohafaza))
        if not os.path.exists(output_folder + '/%s_%s/' % (service_name, mohafaza)):
            os.makedirs(output_folder + '/%s_%s/' % (service_name, mohafaza))
        df_test_curr.to_csv(output_folder + '%s_%s/test_%s.csv' % (service_name, mohafaza, model_name))

    return df_test_curr


# def mean_absolute_percentage_error(y_true, y_pred):
#     '''
#     Function to compute the mean absolute percentage error (MAPE) between an actual and
#     predicted vectors
#     :param y_true: the actual values
#     :param y_pred: the predicted values
#     :return: MAPE
#     '''
#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_stats(y_test, y_pred, nb_columns):
    '''
    Function to compute regression error metrics between actual and predicted values +
    correlation between both using different methods: Pearson, Spearman, and Distance
    :param y_test: the actual values. Example df['actual'] (the string inside is the name
    of the actual column. Example: df['LE (mm)'], df['demand'], etc.)
    :param y_pred: the predicted vlaues. Example df['predicted']
    :param nb_columns: number of columns <<discarding the target variable column>>
    :return: R2, Adj-R2, RMSE, MSE, MAE, MAPE
    '''

    if not isinstance(y_test, list):
        y_test = list(y_test)
    if not isinstance(y_pred, list):
        y_pred = list(y_pred)

    n = len(y_test)
    r2_Score = r2_score(y_test, y_pred) # r-squared
    adjusted_r2 = 1 - ((1 - r2_Score) * (n - 1)) / (n - nb_columns - 1) # adjusted r-squared
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred)) # RMSE
    mse_score = mean_squared_error(y_test, y_pred) # MSE
    mae_score = mean_absolute_error(y_test, y_pred) # MAE
    mape_score = mean_absolute_percentage_error(y_test, y_pred) # MAPE

    if len(y_test) >= 2:
        pearson_corr, _ = pearsonr(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        distance_corr = distance.correlation(y_test, y_pred)

        return r2_Score, adjusted_r2, rmse_score, mse_score, mae_score, mape_score, pearson_corr, spearman_corr, distance_corr
    else:

        return r2_Score, adjusted_r2, rmse_score, mse_score, mae_score, mape_score


def replace_zeros_with_ones(df):
    df = df.replace({'demand': {0: 1}})
    lag_cols = ['w_{t-1}', 'w_{t-2}', 'w_{t-3}', 'w_{t-4}', 'w_{t-5}']
    for col in lag_cols:
        if col in df.columns:
            df = df.replace({col: {0: 1}})
    return df


def check_service_mohafaza(file_name):
    services = ['General Medicine', 'Gynaecology', 'Pediatrics', 'Pharmacy']
    mohafazas = ['akkar', 'bikaa', 'Tripoli']
    curr_service, curr_mohafaza, curr_datasubset = None, None, None
    for service in services:
        for mohafaza in mohafazas:
            # if the service and mohafaza are substring of the file's path
            if service in file_name and mohafaza in file_name:
                curr_service = service
                curr_mohafaza = mohafaza
                curr_datasubset = '%s_%s' % (curr_service, curr_mohafaza)
                print('reading %s in %s data subset ... ' % (curr_service, curr_mohafaza))

    return curr_service, curr_mohafaza, curr_datasubset