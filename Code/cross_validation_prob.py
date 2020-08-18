from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.spatial import distance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import time
import itertools as it
from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
import pickle
from sklearn.compose import ColumnTransformer
from scipy.stats.stats import pearsonr, spearmanr
import scipy.special
import scipy.stats
from Code.probabilistic_uncertainty import mc_dropout_model, deep_ensemble_training
from Code.probabilistic_uncertainty.bootstrap_evaluator import bootstrap_evaluation
from Code.probabilistic_uncertainty.combined_evaluator import combined_evaluation
from Code.probabilistic_uncertainty.mixture_evaluator import mixture_evaluation

from ngboost import NGBRegressor
import tensorflow as tf
from operator import itemgetter
import random


class ProbabilisticForecastsAnalyzer:

    def __init__(self, df, target_variable, split_ratio: float,
                    output_folder,
                    cols_drop=None, scale=True, scale_input=True, scale_output=False,
                    output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                    input_zscore=None, input_minmax=None, input_box=None, input_log=None,
                    testing_data=None,
                    grid=True, random_grid=False,
                    nb_folds_grid=None, nb_repeats_grid=None,
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
        self.split_ratio = split_ratio

        # save error metrics to xlsx sheet
        self.save_errors_xlsx = save_errors_xlsx
        self.save_validation = save_validation
        if self.save_errors_xlsx:
            # data frame in case of probabilistic forecasts
            self.results = pd.DataFrame(columns=[
                'r2', 'adj_r2', 'rmse', 'mse', 'mae', 'mape',
                'avg_%s' % self.target_variable,
                'nll', 'loss',
                'pearson', 'spearman', 'distance',
                'winning_hyperparams', 'training_time_min', 'training_time_sec'
            ])
        else:
            # self.results = None
            self.results = None

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

        # dictionary containing NLL scores of each model
        self.nlls = {}
        self.uncertainties = {}

    def save_train_test_before_modeling(self):
        ''' save the training and testing data frames before any processing happens to them '''
        path = self.output_folder + 'train_test_before_modeling/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.df_train.to_csv(path + 'training.csv', index=False)
        self.df_test.to_csv(path + 'testing.csv', index=False)

    def mkdir(self, dir):
        ''' creates directory if it does not exist '''
        if not os.path.exists(dir):
            os.makedirs(dir)

    def plot_probas(self, ngboost_tr, X_test, y_test, cands, output_folder, fig_name):
        '''
        :param ngboost_tr: trained ng boost model
        :param X_test: testing input data
        :param cands: candidates cherry picked before hand - indices inside the testing data
        :return:
        '''
        cmap = self.get_cmap(len(cands))
        colors = [cmap(i) for i in range(len(cands))]
        fig, ax = plt.subplots(1, 1)
        for cand, c in zip(cands, colors):
            y_pred_dist = ngboost_tr.pred_dist(X_test[cand, :].reshape(1, -1))
            # x_span = np.linspace(y_pred_dist.ppf(0.01)[0], y_pred_dist.ppf(0.99)[0], 100)
            x_span = np.linspace(min(y_test), max(y_test), 100)
            dist_values = y_pred_dist.pdf(x_span)
            lab = y_pred_dist.mean()[0]
            ax.plot(x_span, dist_values, color=c, label=int(lab))
            ax.legend(loc="upper right")
            del y_pred_dist, x_span, dist_values

        plt.xlabel('{} values'.format(self.target_variable))
        plt.ylabel('probability')
        plt.title('Predicted Probability Distributions on {} Random Rare Values'.format(len(cands)))

        self.mkdir(output_folder)

        plt.savefig(os.path.join(output_folder, fig_name))
        plt.close()

    def plot_errors_mc_dropout(self, Yt_hat, X, y, T, output_folder, fig_name):
        pred = np.zeros((X.shape[0], T))  # empty array to be populated
        means = []  # save mean for each predicted point
        std = []  # save standard dev for each predicted
        for j in range(X.shape[0]):
            for i in range(T):
                pred[j][i] = Yt_hat[i][j]
            means.append(pred[j].mean())  # get the mean for each prediction
            std.append(pred[j].std())  # get the standard deviation

        plt.figure(figsize=(16, 16))  # make the size of the plot a bit bigger
        plt.errorbar(x=list(range(X.shape[0])), y=means, yerr=std, fmt='x', label='errors')
        plt.scatter(list(range(X.shape[0])), y, c='r',
                    label='real')  # add the real values on top with red color
        plt.legend(loc='best')

        self.mkdir(output_folder)

        plt.savefig(os.path.join(output_folder, fig_name))
        plt.close()

    def cross_validation_uncertainties(self, possible_hyperparams, model_name):

        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        # training and testing data if we have single target variable
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        tempModels = []

        # specify the type of cv (kfold vs. repeated kfold)
        if self.nb_repeats_grid is None:
            print('running %d-fold cross validation' % self.nb_folds_grid)
            kf = KFold(n_splits=self.nb_folds_grid, random_state=2652124)
        else:
            print('running %d-fold-%d-repeats cross validation' % (self.nb_folds_grid, self.nb_repeats_grid))
            kf = RepeatedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid, random_state=2652124)

        t1 = time.time()
        parameters = possible_hyperparams

        # hyper parameters loop
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []
            uncertainties, epis_uncertainties, ale_uncertainties, tot_uncertainties = [], [], [], []
            for train_index, test_index in kf.split(X_train):

                tf.reset_default_graph()

                X_train_inner, X_val = X_train[train_index], X_train[test_index]
                y_train_inner, y_val = y_train[train_index], y_train[test_index]

                drp = parameter['dropout']
                lr = parameter['learning_rate']
                epochs = parameter['epochs']

                if self.scale:
                    X_train_inner, X_val, y_train_inner, y_val, scaler_out_final = self.scale_cols(X_train_inner, X_val,
                                                                                                   y_train_inner, y_val)

                if model_name == 'bootstrap':
                    n_heads = parameter['n_heads']
                    # does not differ between epistemic and aleatoric uncertainty
                    y_pred, uncertainty_v, loss = bootstrap_evaluation(X_train=X_train_inner,
                                                                 y_train=y_train_inner,
                                                                 X_test=X_val,
                                                                 y_test=y_val,
                                                                 dropout=drp,
                                                                 learning_rate=lr, epochs=epochs, n_heads=n_heads)

                    uncertainties.append(np.mean(uncertainty_v))

                elif model_name == 'combined':
                    n_passes = parameter['n_passes']
                    y_pred, epistemic_unc_v, aleatoric_unc_v, total_unc_v, loss = combined_evaluation(X_train=X_train_inner,
                                                                                                y_train=y_train_inner,
                                                                                                X_test=X_val,
                                                                                                y_test=y_val,
                                                                                                dropout=drp,
                                                                                                learning_rate=lr, epochs=epochs,
                                                                                                n_passes=n_passes)
                    epis_uncertainties.append(np.mean(epistemic_unc_v))
                    ale_uncertainties.append(np.mean(aleatoric_unc_v))
                    tot_uncertainties.append(np.mean(total_unc_v))
                else:
                    # the mixture
                    n_mixtures = parameter['n_mixtures']
                    y_pred, epistemic_unc_v, aleatoric_unc_v, total_unc_v, loss = \
                        mixture_evaluation(X_train=X_train_inner, y_train=y_train_inner, X_test=X_val, y_test=y_val,
                                           dropout=drp, learning_rate=lr, epochs=epochs, n_mixtures=n_mixtures)

                    epis_uncertainties.append(np.mean(epistemic_unc_v))
                    ale_uncertainties.append(np.mean(aleatoric_unc_v))
                    tot_uncertainties.append(np.mean(total_unc_v))

                r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = self.get_stats(y_val, y_pred,
                                                                                               X_val.shape[1])
                # add the point prediction error metrics
                r2_scores.append(r2)
                adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)

            # no aleatoric and epistemic uncertainties
            if model_name == 'bootstrap':
                tempModels.append(
                    [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores),
                     np.mean(mse_scores),
                     np.mean(mae_scores), np.mean(mape_scores), np.mean(uncertainties)])

            # with aleatoric and epistemic uncertainties
            else:
                # print('epis_uncertainties.shape: {}'.format(epis_uncertainties.shape))
                # print('ale_uncertainties.shape: {}'.format(ale_uncertainties.shape))
                tempModels.append(
                    [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores),
                     np.mean(mse_scores),
                     np.mean(mae_scores), np.mean(mape_scores),
                     np.mean(epis_uncertainties), np.mean(ale_uncertainties),
                     np.mean(tot_uncertainties)])


        tf.reset_default_graph()

        tempModels = sorted(tempModels, key=lambda k: k[3])
        winning_hyperparameters = tempModels[0][0]

        print('winning hyper parameters: ', str(winning_hyperparameters))
        print(
            'Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
            (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
             tempModels[0][6]))

        if self.save_errors_xlsx:
            if self.save_validation:
                self.results.loc['{}_val'.format(model_name)] = pd.Series({'r2': tempModels[0][1],
                                                                           'adj_r2': tempModels[0][2],
                                                                           'rmse': tempModels[0][3],
                                                                           'mse': tempModels[0][4],
                                                                           'mae': tempModels[0][5],
                                                                           'mape': tempModels[0][6]})

        if model_name == 'bootstrap':
            self.uncertainties[model_name] = {
                'val_epistemic_unc': 0,
                'val_aleatoric_unc': 0,
                'val_total_unc': tempModels[0][7]
            }
        else:
            self.uncertainties[model_name] = {
                'val_epistemic_unc': tempModels[0][7],
                'val_aleatoric_unc': tempModels[0][8],
                'val_total_unc': tempModels[0][9]
            }

        drp = winning_hyperparameters['dropout']
        lr = winning_hyperparameters['learning_rate']
        epochs = winning_hyperparameters['epochs']
        # loss = None

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        if model_name == 'bootstrap':
            n_heads = winning_hyperparameters['n_heads']
            # does not differ between epistemic and aleatoric uncertainty
            y_pred, uncertainty, loss = bootstrap_evaluation(X_train=X_train, y_train=y_train,
                                                       X_test=X_test, y_test=y_test,
                                                       dropout=drp, learning_rate=lr, epochs=epochs,
                                                                            n_heads=n_heads)
        elif model_name == 'combined':
            n_passes = winning_hyperparameters['n_passes']
            y_pred, epistemic_unc, aleatoric_unc, total_unc, loss = combined_evaluation(X_train=X_train, y_train=y_train,
                                                                                  X_test=X_test, y_test=y_test,
                                                                                  dropout=drp, learning_rate=lr,
                                                                                  epochs=epochs,
                                                                                  n_passes=n_passes)
        else:
            # the mixture
            n_mixtures = winning_hyperparameters['n_mixtures']
            y_pred, epistemic_unc, aleatoric_unc, total_unc, loss = mixture_evaluation(X_train=X_train, y_train=y_train,
                                                                                 X_test=X_test,
                                                                                 y_test=y_test,
                                                                                 dropout=drp, learning_rate=lr,
                                                                                 epochs=epochs,
                                                                                 n_mixtures=n_mixtures)

        t2 = time.time()
        time_taken_min = float(t2 - t1) / 60
        time_taken_sec = float(t2 - t1)

        self.plot_actual_vs_predicted(y_test, y_pred, self.output_folder + 'train_test_forecasts_lineplot/',
                                      model_name)

        self.plot_actual_vs_predicted_scatter_bisector(y_test, y_pred,
                                                       self.output_folder + 'train_test_forecasts_scatterplot_bisector/',
                                                       model_name)

        r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = self.get_stats(y_test, y_pred, X_test.shape[1])
        print('Testing Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape))

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))
        print('Pearson Correlation: %.5f' % pearson)
        print('Spearman Correlation: %.5f' % spearman)
        print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        if self.save_errors_xlsx:
            if self.save_validation:
                row_name = '{}_test'.format(model_name)
            else:
                row_name = model_name
            print('Saving results to csv file ...')

            self.results.loc[row_name] = pd.Series({'r2': r2, 'adj_r2': adj_r2,
                                                    'rmse': rmse, 'mse': mse,
                                                    'mae': mae, 'mape': mape,
                                                    # 'nll': '-' if loss is None else loss,
                                                    'loss': loss,
                                                    'avg_%s' % self.target_variable: avg_target,
                                                    'pearson': pearson, 'spearman': spearman,
                                                    'distance': distance,
                                                    'winning_hyperparams': str(winning_hyperparameters),
                                                    'training_time_min': time_taken_min,
                                                    'training_time_sec': time_taken_sec
                                                    })

        if model_name == 'bootstrap':
            self.uncertainties[model_name]['test_epistemic_unc'] = 0
            self.uncertainties[model_name]['test_aleatoric_unc'] = 0
            self.uncertainties[model_name]['test_total_unc'] = np.mean(uncertainty)

        else:
            self.uncertainties[model_name]['test_epistemic_unc'] = np.mean(epistemic_unc)
            self.uncertainties[model_name]['test_aleatoric_unc'] = np.mean(aleatoric_unc)
            self.uncertainties[model_name]['test_total_unc'] = np.mean(total_unc)

    def cross_validation_deep_ensemble(self, possible_hyperparams, sort_by='rmse'):
        model_name = 'deep_ensemble'

        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        # training and testing data if we have single target variable
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        tempModels = []

        # specify the type of cv (kfold vs. repeated kfold)
        if self.nb_repeats_grid is None:
            print('running %d-fold cross validation' % self.nb_folds_grid)
            kf = KFold(n_splits=self.nb_folds_grid, random_state=2652124)
        else:
            print('running %d-fold-%d-repeats cross validation' % (self.nb_folds_grid, self.nb_repeats_grid))
            kf = RepeatedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid, random_state=2652124)

        t1 = time.time()
        parameters = possible_hyperparams

        # hyper parameters loop
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):

            # additional lists specific to ng boost
            # references: https://github.com/stanfordmlgroup/ngboost/blob/master/examples/experiments/regression_exp.py
            de_nll = []
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []
            uncertainties = []

            for train_index, test_index in kf.split(X_train):

                tf.reset_default_graph()

                X_train_inner, X_val = X_train[train_index], X_train[test_index]
                y_train_inner, y_val = y_train[train_index], y_train[test_index]

                lr = parameter['learning_rate']
                bs = parameter['batch_size']
                epochs = parameter['max_iter']
                layers = parameter['sizes']
                opt = parameter['optimizer_name']

                if self.scale:
                    X_train_inner, X_val, y_train_inner, y_val, scaler_out_final = self.scale_cols(X_train_inner, X_val,
                                                                                                   y_train_inner, y_val)

                # mean, var, nll, outputs_per_ensemble
                y_pred, var, val_nll, _ = deep_ensemble_training.train_ensemble(X_train_inner, y_train_inner, X_val, y_val, learning_rate=lr,
                                                     max_iter=epochs, batch_size=bs,
                                                     sizes=layers, optimizer_name=opt,
                                                     produce_plots=False)

                # error metrics for the regular point predictions
                r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = self.get_stats(y_val, y_pred,
                                                                                               X_val.shape[1])

                # add the point prediction error metrics
                r2_scores.append(r2)
                adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)

                # add the ngb probabilistic predictions
                de_nll.append(val_nll)

                # add the uncertainties
                uncertainties.append(np.mean(var))

            tempModels.append(
                [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores), np.mean(mse_scores),
                 np.mean(mae_scores), np.mean(mape_scores),
                 np.mean(de_nll), np.std(de_nll), np.mean(uncertainties)])

        tf.reset_default_graph()

        # the best by RMSE
        if sort_by == 'rmse':
            tempModels = sorted(tempModels, key=lambda k: k[3])
            winning_hyperparameters = tempModels[0][0]
        else:
            tempModels = sorted(tempModels, key=lambda k: k[7])
            winning_hyperparameters = tempModels[0][0]

        # store the mean and standard deviation
        # of the NLL in cross validation phase of the
        # winning hyper parameter only (no point in storing those of all hyper parameters)
        self.nlls['deep_ensemble'] = {
            'mean': tempModels[0][7],
            'std': tempModels[0][8]
        }

        self.uncertainties[model_name] = {
            'val_epistemic_unc': 0,
            'val_aleatoric_unc': 0,
            'val_total_unc': tempModels[0][9]
        }

        print('winning hyper parameters: ', str(winning_hyperparameters))
        print(
            'Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\nDE-NLL: %.5f\n' %
            (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
             tempModels[0][6], tempModels[0][7]))

        if self.save_errors_xlsx:
            if self.save_validation:
                self.results.loc['{}_val'.format(model_name)] = pd.Series({'r2': tempModels[0][1],
                                                        'adj_r2': tempModels[0][2],
                                                        'rmse': tempModels[0][3],
                                                        'mse': tempModels[0][4],
                                                        'mae': tempModels[0][5],
                                                        'mape': tempModels[0][6],
                                                        'nll': tempModels[0][7]})

        lr = winning_hyperparameters['learning_rate']
        bs = winning_hyperparameters['batch_size']
        epochs = winning_hyperparameters['max_iter']
        layers = winning_hyperparameters['sizes']
        opt = winning_hyperparameters['optimizer_name']

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        # mean, var, nll, outputs_per_ensemble
        y_pred, var, test_nll, _ = deep_ensemble_training.train_ensemble(X_train, y_train, X_test, y_test, learning_rate=lr,
                                                 max_iter=epochs, batch_size=bs,
                                                 sizes=layers, optimizer_name=opt,
                                                 produce_plots=True,
                                                 output_folder= os.path.join(self.output_folder, 'probabilistic_forecasts/'),
                                                 fig_name='deep_ensemble')

        self.uncertainties[model_name]['test_epistemic_unc'] = 0
        self.uncertainties[model_name]['test_aleatoric_unc'] = 0
        self.uncertainties[model_name]['test_total_unc'] = np.mean(var)

        t2 = time.time()
        time_taken_min = float(t2 - t1) / 60
        time_taken_sec = float(t2 - t1)

        if np.any(np.isnan(test_nll)):
            raise ValueError('Oops, your testing NLL is nan')

        self.plot_actual_vs_predicted(y_test, y_pred, self.output_folder + 'train_test_forecasts_lineplot/',
                                      model_name)

        self.plot_actual_vs_predicted_scatter_bisector(y_test, y_pred, self.output_folder + 'train_test_forecasts_scatterplot_bisector/',
                                                       model_name)

        r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = self.get_stats(y_test, y_pred, X_test.shape[1])
        print('Testing Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape))

        print('DE-NLL: %.5f' % test_nll)

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))
        print('Pearson Correlation: %.5f' % pearson)
        print('Spearman Correlation: %.5f' % spearman)
        print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        if self.save_errors_xlsx:
            if self.save_validation:
                row_name = '{}_test'.format(model_name)
            else:
                row_name = model_name
            print('Saving results to csv file ...')

            self.results.loc[row_name] = pd.Series({'r2': r2, 'adj_r2': adj_r2,
                                                    'rmse': rmse, 'mse': mse,
                                                    'mae': mae, 'mape': mape,
                                                    'nll': test_nll,
                                                    'avg_%s' % self.target_variable: avg_target,
                                                    'pearson': pearson, 'spearman': spearman,
                                                    'distance': distance,
                                                    'winning_hyperparams': str(winning_hyperparameters),
                                                    'training_time_min': time_taken_min,
                                                    'training_time_sec': time_taken_sec
                                                    })
            self.nlls['deep_ensemble']['test'] = test_nll

    def cross_validation_grid_ngboost(self, possible_hyperparams, sort_by='rmse'):
        model_name = 'ngboost'

        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        # training and testing data if we have single target variable
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        tempModels = []

        # specify the type of cv (kfold vs. repeated kfold)
        if self.nb_repeats_grid is None:
            print('running %d-fold cross validation' % self.nb_folds_grid)
            kf = KFold(n_splits=self.nb_folds_grid, random_state=2652124)
        else:
            print('running %d-fold-%d-repeats cross validation' % (self.nb_folds_grid, self.nb_repeats_grid))
            kf = RepeatedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid, random_state=2652124)

        t1 = time.time()

        parameters = possible_hyperparams

        # hyper parameters loop
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):

            model = NGBRegressor(**parameter)

            # additional lists specific to ng boost
            # references: https://github.com/stanfordmlgroup/ngboost/blob/master/examples/experiments/regression_exp.py
            ngb_nll = []
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []

            for train_index, test_index in kf.split(X_train):
                X_train_inner, X_val = X_train[train_index], X_train[test_index]
                y_train_inner, y_val = y_train[train_index], y_train[test_index]

                if self.scale:
                    X_train_inner, X_val, y_train_inner, y_val, scaler_out_final = self.scale_cols(X_train_inner, X_val,
                                                                                                   y_train_inner, y_val)

                model.fit(X_train_inner, y_train_inner)

                y_pred = model.predict(X_val)
                y_forecast = model.pred_dist(X_val)

                # AFTER PREDICTION, reverse the scaled output
                # (if self.scale_output is on). The idea is to reverse scaling
                # JUST BEFORE printing out the error metrics
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

                # error metrics for the regular point predictions
                r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = self.get_stats(y_val, y_pred,
                                                                                               X_val.shape[1])

                # add the point prediction error metrics
                r2_scores.append(r2)
                adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)

                # add the ngb probabilistic predictions
                ngb_nll.append(-y_forecast.logpdf(y_val.flatten()).mean())

            tempModels.append(
                [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores), np.mean(mse_scores),
                 np.mean(mae_scores), np.mean(mape_scores),
                 np.mean(ngb_nll), np.std(ngb_nll)])

        # the best by RMSE
        if sort_by == 'rmse':
            tempModels = sorted(tempModels, key=lambda k: k[3])
            winning_hyperparameters = tempModels[0][0]
        else:
            # sort by the best NGB RMSE
            tempModels = sorted(tempModels, key=lambda k: k[7])
            winning_hyperparameters = tempModels[0][0]

        # store the mean and standard deviation
        # of the NLL in cross validation phase of the
        # winning hyper parameter only (no point in storing those of all hyper parameters)
        self.nlls['ng_boost'] = {
            'mean': tempModels[0][7],
            'std': tempModels[0][8]
        }

        print('winning hyper parameters: ', str(winning_hyperparameters))
        print(
            'Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\nNGB-NLL: %.5f +/- %.5f\n' %
            (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
             tempModels[0][6], tempModels[0][7], tempModels[0][8]))

        # store only the mean of nll
        if self.save_errors_xlsx:
            if self.save_validation:
                self.results.loc['{}_val'.format(model_name)] = pd.Series({'r2': tempModels[0][1],
                                                                     'adj_r2': tempModels[0][2],
                                                                     'rmse': tempModels[0][3],
                                                                     'mse': tempModels[0][4],
                                                                     'mae': tempModels[0][5],
                                                                     'mape': tempModels[0][6],
                                                                     'ngb_nll': tempModels[0][7]})

        # re-initialize model with the best set of hyper parameters
        model = NGBRegressor(**winning_hyperparameters)

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        # train the model
        model.fit(X_train, y_train)

        # save the model
        models_folder = self.output_folder + 'trained_models/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        pkl_filename = "{}.pkl".format(model_name)
        with open(models_folder + pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        print('saved model to {} as {}.pkl'.format(models_folder, model_name))

        # the point and probabilistic predictions
        y_pred = model.predict(X_test)
        forecast = model.pred_dist(X_test)

        # the NGB NLL
        ngb_nll = -forecast.logpdf(y_test.flatten()).mean()

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
            print('created output dataset')
        else:
            output_dataset = self.create_output_dataset(y_pred, model_name,
                                                        self.output_folder + 'output_vector_datasets/')
            print('created output dataset')

        print('creating plots ...')

        candidates = np.random.choice(len(X_test), 5)
        self.plot_probas(model, X_test, y_test, candidates, output_folder=self.output_folder + 'probabilistic_forecasts/',
                         fig_name='ngboost_probas_rand')
        self.plot_actual_vs_predicted(y_test, y_pred,
                                      self.output_folder + 'train_test_forecasts_lineplot/',
                                      model_name)

        self.plot_actual_vs_predicted_scatter_bisector(y_test, y_pred,
                                      self.output_folder + 'train_test_forecasts_scatterplot_bisector/',
                                      model_name)

        r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = self.get_stats(y_test, y_pred, X_test.shape[1])
        print('Testing Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape))

        print('NGB-NLL: %.5f' % ngb_nll)

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))
        print('Pearson Correlation: %.5f' % pearson)
        print('Spearman Correlation: %.5f' % spearman)
        print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        if self.save_errors_xlsx:
            if self.save_validation:
                row_name = '{}_test'.format(model_name)
            else:
                row_name = model_name
            print('Saving results to csv file ...')

            self.results.loc[row_name] = pd.Series({'r2': r2, 'adj_r2': adj_r2,
                                                        'rmse': rmse, 'mse': mse,
                                                        'mae': mae, 'mape': mape,
                                                        'nll': ngb_nll,
                                                        'avg_%s' % self.target_variable: avg_target,
                                                        'pearson': pearson, 'spearman': spearman,
                                                        'distance': distance,
                                                        'winning_hyperparams': str(winning_hyperparameters),
                                                        'training_time_min': time_taken_min,
                                                        'training_time_sec': time_taken_sec
                                                    })

            # add the nll on the testing data
            self.nlls['ng_boost']['test'] = ngb_nll

        if not os.path.exists(self.output_folder + '/winning_hyperparams/'):
            os.makedirs(self.output_folder + '/winning_hyperparams/')
        with open(os.path.join(self.output_folder, 'winning_hyperparams/{}_hyperparams.pickle'.format(model_name)), 'wb') as handle:
            pickle.dump(winning_hyperparameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def cross_validation_grid_mc_dropout(self, possible_hyperparams, sort_by='rmse'):

        model_name = 'mc_dropout'

        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        tempModels = []

        if self.nb_repeats_grid is None:
            print('running %d-fold cross validation' % self.nb_folds_grid)
            kf = KFold(n_splits=self.nb_folds_grid, random_state=2652124)
        else:
            print('running %d-fold-%d-repeats cross validation' % (self.nb_folds_grid, self.nb_repeats_grid))
            kf = RepeatedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid, random_state=2652124)

        t1 = time.time()

        parameters = possible_hyperparams

        # hyper parameters loop
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):

            mc_nll = []
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []
            uncertainties = []

            for train_index, test_index in kf.split(X_train):
                X_train_inner, X_val = X_train[train_index], X_train[test_index]
                y_train_inner, y_val = y_train[train_index], y_train[test_index]

                # get the hyper parameters
                n_hidden = parameter['n_hidden']
                num_hidden_layers = parameter['num_hidden_layers']
                n_epochs = parameter['n_epochs']
                epochs_multiplier = parameter['epochx']
                tau = parameter['tau']
                dropout_rate = parameter['dropout_rate']
                norm = parameter['normalize']

                if self.scale:
                    X_train_inner, X_val, y_train_inner, y_val, scaler_out_final = self.scale_cols(X_train_inner, X_val,
                                                                                                   y_train_inner, y_val)

                model = mc_dropout_model.net(X_train_inner, y_train_inner, ([int(n_hidden)] * num_hidden_layers),
                                       normalize=norm, n_epochs=int(n_epochs * epochs_multiplier), tau=tau,
                                       dropout=dropout_rate)

                y_pred, yt_hat, MC_nll, uncertainty = model.predict(X_val, y_val)

                # print('Validation uncertainty: {}'.format(uncertainty))

                # error metrics for the regular point predictions
                r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = self.get_stats(y_val, y_pred,
                                                                                               X_val.shape[1])

                # add the point prediction error metrics
                r2_scores.append(r2)
                adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)

                # add the probabilistic prediction error metrics
                mc_nll.append(MC_nll)

                uncertainties.append(np.mean(uncertainty))

            tempModels.append(
                [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores), np.mean(mse_scores),
                 np.mean(mae_scores), np.mean(mape_scores),
                 np.mean(mc_nll), np.std(mc_nll), np.mean(uncertainties)])

        # the best by RMSE
        if sort_by == 'rmse':
            tempModels = sorted(tempModels, key=lambda k: k[3])
            winning_hyperparameters = tempModels[0][0]
        else:
            # sort by the best NGB RMSE
            tempModels = sorted(tempModels, key=lambda k: k[7])
            winning_hyperparameters = tempModels[0][0]

        # store the mean and standard deviation
        # of the NLL in cross validation phase of the
        # winning hyper parameter only (no point in storing those of all hyper parameters)
        self.nlls['mc_dropout'] = {
            'mean': tempModels[0][7],
            'std': tempModels[0][8]
        }

        self.uncertainties[model_name] = {
            'val_epistemic_unc': 0,
            'val_aleatoric_unc': 0,
            'val_total_unc': tempModels[0][9]
        }

        if self.save_errors_xlsx:
            if self.save_validation:
                self.results.loc['{}_val'.format(model_name)] = pd.Series({'r2': tempModels[0][1],
                                                                     'adj_r2': tempModels[0][2],
                                                                     'rmse': tempModels[0][3],
                                                                     'mse': tempModels[0][4],
                                                                     'mae': tempModels[0][5],
                                                                     'mape': tempModels[0][6],
                                                                     'nll': tempModels[0][7]})

        print('winning hyper parameters: {}'.format(str(winning_hyperparameters)))
        print(
            'Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\nMC-NLL: %.5f +/- %.5f\n' %
            (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
             tempModels[0][6], tempModels[0][7], tempModels[0][8]))

        # get the winning hyper parameters
        n_hidden = winning_hyperparameters['n_hidden']
        num_hidden_layers = winning_hyperparameters['num_hidden_layers']
        n_epochs = winning_hyperparameters['n_epochs']
        epochs_multiplier = winning_hyperparameters['epochx']
        tau = winning_hyperparameters['tau']
        dropout_rate = winning_hyperparameters['dropout_rate']

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        model = mc_dropout_model.net(X_train, y_train, ([int(n_hidden)] * num_hidden_layers),
                               normalize=True, n_epochs=int(n_epochs * epochs_multiplier), tau=tau,
                               dropout=dropout_rate)

        y_pred, yt_hat, MC_nll, uncertainty = model.predict(X_test, y_test)

        # print('Testing uncertainty: {}'.format(uncertainty))

        self.uncertainties[model_name]['test_epistemic_unc'] = 0
        self.uncertainties[model_name]['test_aleatoric_unc'] = 0
        self.uncertainties[model_name]['test_total_unc'] = np.mean(uncertainty)

        t2 = time.time()
        time_taken_min = float(t2 - t1) / 60
        time_taken_sec = float(t2 - t1)

        # average of the `actual` target variable
        avg_target = np.mean(y_test)

        # point prediction errors
        r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance = self.get_stats(y_test, y_pred,
                                                                                           X_test.shape[1])

        # display error metrics
        print('Testing Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape))

        print('MC-NLL: %.5f' % MC_nll)

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))
        print('Pearson Correlation: %.5f' % pearson)
        print('Spearman Correlation: %.5f' % spearman)
        print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        # save the model
        models_folder = self.output_folder + 'trained_models/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        pkl_filename = "{}.pkl".format(model_name)
        with open(models_folder + pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        print('saved model to {} as {}.pkl'.format(models_folder, model_name))

        # save results to csv error files
        self.results.loc[model_name] = pd.Series({
            'r2': r2, 'adj_r2': adj_r2,
            'rmse': rmse, 'mse': mse,
            'mae': mae, 'mape': mape,
            'nll': MC_nll,
            'nb_splits': None,
            'avg_%s' % self.target_variable: avg_target,
            'pearson': pearson, 'spearman': spearman,
            'distance': distance,
            'winning_hyperparams': str(winning_hyperparameters),
            'training_time_min': time_taken_min,
            'training_time_sec': time_taken_sec
        })

        # add the nll on the testing data
        self.nlls['mc_dropout']['test'] = MC_nll

        # Plotting
        self.plot_actual_vs_predicted(y_test, y_pred, self.output_folder + 'train_test_forecasts_lineplot/',
                                      model_name)
        self.plot_actual_vs_predicted_scatter_bisector(y_test, y_pred,
                                                       self.output_folder + 'train_test_forecasts_scatterplot_bisector/',
                                                       model_name)

        self.plot_errors_mc_dropout(Yt_hat=yt_hat, X=X_test, y=y_pred, T=winning_hyperparameters['T'],
                         output_folder=os.path.join(self.output_folder, 'probabilistic_forecasts/'),
                         fig_name='{}_errors'.format(model_name))

        if not os.path.exists(self.output_folder + '/winning_hyperparams/'):
            os.makedirs(self.output_folder + '/winning_hyperparams/')
        with open(os.path.join(self.output_folder, 'winning_hyperparams/{}_hyperparams.pickle'.format(model_name)), 'wb') as handle:
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

        # df_train.to_csv('train_df.csv')
        df_test_curr.to_csv(output_folder + 'test_df_%s.csv' % model_name, index=False)

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

    def plot_actual_vs_predicted(self, y_test, y_pred, output_folder, model_name):
        plt.plot(list(range(len(y_test))), y_test, color='b', label='actual')
        plt.plot(list(range(len(y_pred))), y_pred, color='r', label='predicted')
        plt.legend(loc='best')
        plt.suptitle('actual vs. predicted forecasts')

        self.mkdir(output_folder)
        plt.savefig(output_folder + 'forecasts_{}'.format(model_name))
        plt.close()

    def plot_actual_vs_predicted_scatter_bisector(self, y_test, y_pred, output_folder, model_name):
        fig, ax = plt.subplots()

        ax.scatter(y_test, y_pred, c='black')

        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        plt.suptitle('actual vs. predicted forecasts')

        self.mkdir(output_folder)
        plt.savefig(os.path.join(output_folder, 'scatter_%s' % model_name))
        plt.close()

    def save_uncertainties(self):
        dest = os.path.join(self.output_folder, 'uncertainty_scores')
        self.mkdir(dest)
        with open(os.path.join(dest, 'uncertainty_scores.p'), 'wb') as fp:
            pickle.dump(self.uncertainties, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def save_nlls(self):
        '''
        saves the NLL scores of each probabilistic model - dict of dicts
        KEY: mean: contains the mean NLL of cross validation with winning hyper params
        KEY: std: contains the std NLL of cross validation with winning hyper params
        KEY: test: contains the NLL score on the testing data
        '''
        dest = os.path.join(self.output_folder, 'nll_scores')
        self.mkdir(dest)
        with open(os.path.join(dest, 'nll_scores.p'), 'wb') as fp:
            pickle.dump(self.nlls, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def show_nll_scores(self):
        ''' error bars from cv and testing NLLs among different models '''
        lists = []
        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(len(self.nlls))]
        i = 1
        counter = 0
        for model in self.nlls:
            temp_list = []
            temp_list.append(model)

            mean_nll = self.nlls[model]['mean']
            std_nll = self.nlls[model]['std']
            test_nll = self.nlls[model]['test']

            temp_list.append(mean_nll)
            temp_list.append(std_nll)
            temp_list.append(test_nll)

            plt.errorbar(np.array(i), mean_nll, yerr=std_nll, fmt='o', c=color[counter], label='{} cv'.format(model),
                         mec='black')
            plt.scatter(np.array(i + 1), test_nll, c=color[counter], label='{} test'.format(model))

            i = (i*2) + 1
            counter += 1
            lists.append(temp_list)

        plt.legend(loc='lower left', bbox_to_anchor=(0., 1., 1., .102),
          ncol=3, mode='expand', shadow=True, fancybox=True, borderaxespad=0.)

        plt.xticks([])
        plt.xlabel('models')
        plt.ylabel('NLL')
        dest = os.path.join(self.output_folder, 'probabilistic_forecasts/')
        if not os.path.exists(dest):
            os.makedirs(dest)
        plt.savefig(os.path.join(dest, 'nll_cv_test_scores'))
        plt.close()

        # sort list of lists by ascending order of nll (the testing)
        lists = sorted(lists, key=itemgetter(3))
        df = pd.DataFrame(lists, columns=['model', 'cv_mean nll', 'cv_std nll', 'test nll'])
        df.to_csv(os.path.join(dest, 'nlls.csv'), index=False)

    def show_uncertainty_scores(self):
        ''' bar plots from cv and testing scores of uncertaintied across different models '''

        # if dictionary is not empty
        if self.uncertainties:
            lists = []
            for model in self.uncertainties:
                temp_list = []
                temp_list.append(model)

                # validation and testing epistemic uncertainty
                temp_list.append(self.uncertainties[model]['val_epistemic_unc'])
                temp_list.append(self.uncertainties[model]['test_epistemic_unc'])

                # validation and testing aleatoric uncertainty
                temp_list.append(self.uncertainties[model]['val_aleatoric_unc'])
                temp_list.append(self.uncertainties[model]['test_aleatoric_unc'])

                # validation and testing total uncertainty
                temp_list.append(self.uncertainties[model]['val_total_unc'])
                temp_list.append(self.uncertainties[model]['test_total_unc'])
                lists.append(temp_list)

            # sort list of lists by ascending order of uncertainty (the testing)
            lists = sorted(lists, key=itemgetter(6))

            df = pd.DataFrame(lists, columns=['model',
                                              'val epist', 'test epist',
                                              'val aleat', 'test aleat',
                                              'val uncertainty', 'test uncertainty'])

            # df = df.set_index('model')
            df_temp = df
            df_temp = df_temp.set_index('model')
            df_temp[['val uncertainty', 'test uncertainty']].plot(kind='bar', title='uncertainties', legend=True)
            plt.xticks(fontsize=8, rotation=45)
            plt.ylabel('uncertainty')
            plt.xlabel('models')

            # plt.show()
            output_folder = os.path.join(self.output_folder, 'probabilistic_forecasts/')
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            plt.savefig(os.path.join(output_folder, 'uncertainties'))
            plt.close()

            # save the dataset (not the copy)
            df = df.set_index('model')
            df.to_csv(os.path.join(output_folder, 'uncertainties.csv'))

    def errors_to_csv(self):
        ''' saves the error metrics (stored in `results`) as a csv file and the NLL scores '''

        # saving the error metrics
        if self.results is not None:
            errors_df = self.results
            path = self.output_folder + 'error_metrics_csv/'
            if not os.path.exists(path):
                os.makedirs(path)
            errors_df.to_csv(path + 'errors.csv')

        # saving the NLL scores
        self.save_nlls()

        # saving the uncertainty scores
        self.save_uncertainties()

        # plot the NLL scores
        self.show_nll_scores()

        # plot the uncertainty scores
        self.show_uncertainty_scores()

    def get_stats(self, y_test, y_pred, nb_columns):
        '''
        Function to compute regression and utility based error metrics between actual and predicted values as well
        as their correlation
        :param y_test: vector of the actual values
        :param y_pred: vector of the predicted values
        :param nb_columns: number of columns <<discarding the target variable column>>
        :return: R2, Adj-R2, RMSE, MSE, MAE, MAPE, pearson, spearman, distance
        '''

        def mean_absolute_percentage_error(y_true, y_pred):
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        if not isinstance(y_test, list):
            y_test = list(y_test)
        if not isinstance(y_pred, list):
            y_pred = list(y_pred)

        n = len(y_test)

        r2_Score = r2_score(y_test, y_pred)  # r-squared
        adjusted_r2 = 1 - ((1 - r2_Score) * (n - 1)) / (n - nb_columns - 1)  # adjusted r-squared
        rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
        mse_score = mean_squared_error(y_test, y_pred)  # MSE
        mae_score = mean_absolute_error(y_test, y_pred)  # MAE
        mape_score = mean_absolute_percentage_error(y_test, y_pred)  # MAPE

        if isinstance(y_pred[0], np.ndarray):
            y_pred_new = [x[0] for x in y_pred]
            y_pred = y_pred_new

        pearson_corr, _ = pearsonr(y_test, y_pred)
        spearman_corr, _ = spearmanr(y_test, y_pred)
        distance_corr = distance.correlation(y_test, y_pred)

        return r2_Score, adjusted_r2, rmse_score, mse_score, mae_score, mape_score, pearson_corr, spearman_corr, distance_corr
