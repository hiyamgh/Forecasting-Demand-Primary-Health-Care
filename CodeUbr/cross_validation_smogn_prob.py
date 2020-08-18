import numpy as np
import time
import pickle
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, RepeatedKFold, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import learning_curve
from sklearn.compose import ColumnTransformer
from scipy.stats.stats import pearsonr, spearmanr
import scipy.stats
import itertools as it
from extract_rare import *
import matplotlib
import matplotlib.pyplot as plt
from Code.probabilistic_uncertainty import mc_dropout_model, deep_ensemble_training
from Code.probabilistic_uncertainty.bootstrap_evaluator import bootstrap_evaluation
from Code.probabilistic_uncertainty.combined_evaluator import combined_evaluation
from Code.probabilistic_uncertainty.mixture_evaluator import mixture_evaluation
import tensorflow as tf
from ngboost import NGBRegressor
from operator import itemgetter
import random


matplotlib.use('Agg')
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.spatial import distance
import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

pandas2ri.activate()
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
# additional
import rpy2.robjects as ro
from rpy2.robjects.conversion import localconverter

# additional

runit = robjects.r
runit['source']('smogn.R')


def convert_cols_int(df, columns):
    '''
    converts the specified columns from type float into type int
    (We need this because we have one hot encoded columns that are of type float rather than int)
    :param df: the dataframe
    :param columns: list of column names
    :return: df with the columns specified converted to int
    '''
    df[columns] = df[columns].astype(int)
    return df


def get_formula(target_variable):
    '''
    gets the formula for passing it to R functions. Example: target_variable ~ col1 + col2 ...
    :param target_variable: the name of the target variable
    :return: R's formula as follows: target_variable ~ other[0] + other[1] + other[2] + other[3] + ...
    '''
    formula = runit.create_formula(target_variable)
    return formula


def get_method(smogn, smoter, randunder, gn):
    if smogn:
        return 'smogn'
    elif smoter:
        return 'smoter'
    elif randunder:
        return 'randunder'
    elif gn:
        return 'gn'
    return 'nosmote'


def shuffle_rarify_data(df_train, df_test, target_variable, method='extremes', extr_type='both', thresh=0.8,
                        coef=1.5, control_pts=None):
    '''
    shuffle: shuffle the data
    rarify: make rare

    - In our problem, we previously had df_train_collated and df_test_collated.
    - df_test_collated was made up of the <last> 20% row of each of the 12 subsets we have - collated together
    - since it had the last rows of each datasubset, all the rare values lie mostly in the testing data because
      these last 20% of each datasubset were kept in their time series format and it happens that they
      denote the year 2016 of each datasubset and the year 2016 had the most peaks in large scale events
      and in demand
    this burdens the performance of our models because no much rare values exist in the training data
    thats why we will:

    1. get df_train and df_test
    2. combine those into 1 dataframe
    3. shuffle the dataframe (so the rare values are not centralized somewhere)
    4. Obtain new df_train and df_test such that:
       * df_train and df_test have equal class distributions between classes: rare and not rare (see below)
       * get the % of rare in the whole dataframe (df_train + df_test)
       * denote df_train by A
       * denote df_test by B
       * denote the whole dataframe by S
       * S = A U B and S has X% rare
       * make A have X% rare
       * make B have X% rare

    :param df_train: the training data frame
    :param df_test: the testing data frame
    :param target_variable: name of the target variable column
    :return: df_train and df_test with equal class distribution between classes: rare and not rare
    '''

    # concatenate both df_train and df_test into one data frame
    df = pd.concat([df_train, df_test])

    # shuffle the data, use random_state to ensure reproducability of our results
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    # get y, reset the index to avoid falsy retrievals by index later on
    y = df[target_variable].reset_index(drop=True)

    # get the indices of the rare values in the combined data frame
    # note that the relevance returned is the relevance of the whole data frame not just the training
    rare_values, phi_params, loss_params, yrel = get_rare(y, method=method, extr_type=extr_type,
                                                          thresh=thresh, coef=coef, control_pts=control_pts)

    # dictionary mapping each value of demand to its relevance
    demandrel = {}
    relvals = np.array(yrel)
    for i, e in enumerate(y):
        if e not in demandrel:
            rel = relvals[i]
            demandrel[e] = rel

    # now we have the indices of the rare values, get their percentage and ensure equal
    # class distribution between rare and not rare

    # percentage of rare values in the whole dataset
    prare = len(rare_values) / len(df)

    # number of rare values in the whole dataset
    numrare = len(rare_values)
    print('number of rare values: {}/{}'.format(numrare, len(df)))

    # number of rare values that must be in each of the train and test
    numraretrain = int(round(prare * len(df_train)))
    numraretest = int(round(prare * len(df_test)))

    print('number of rare that must be in train: {}/{}'.format(numraretrain, len(df_train)))
    print('==> {}%%'.format(numraretrain / len(df_train)))
    print('number of rare that must be in test: {}/{}'.format(numraretest, len(df_test)))
    print('==> {}%%'.format(numraretest / len(df_test)))

    rare_values = sorted(rare_values)
    # print('rare values sorted: {}'.format(rare_values))

    # rare indices partitioned for each of the train and test
    raretrain = rare_values[:numraretrain]
    raretest = rare_values[numraretrain:]

    # get the rows of the rare values, retrieve by indices
    rarerowstrain = df.iloc[raretrain, :].reset_index(drop=True)
    rarerowstest = df.iloc[raretest, :].reset_index(drop=True)

    # number of rows that remain in training if we remove the rare values
    numrowstrain = len(df_train) - len(rarerowstrain)

    # create temporary df that does not include teh rare values
    dftemp = df.drop(df.index[rare_values])

    # concatenate dftrainrare with rarerowstrain
    dftrainrare = dftemp.iloc[:numrowstrain, :]
    dftrainrare = pd.concat([dftrainrare, rarerowstrain]).sample(frac=1, random_state=1).reset_index(drop=True)

    # concatenate dftestrare with rarerowstest
    dftestrare = dftemp.iloc[numrowstrain:, :]
    dftestrare = pd.concat([dftestrare, rarerowstest]).sample(frac=1, random_state=1).reset_index(drop=True)

    # get the relevance of each of the new dftrainrare and dftestrare
    yreltrain = [demandrel[d] for d in dftrainrare[target_variable]]
    yreltest = [demandrel[d] for d in dftestrare[target_variable]]

    # get the modified indices of the rare values in each of the new dftrainrare and dftestrare
    rtrain = get_rare_indices(dftrainrare[target_variable], yreltrain, thresh, phi_params['control.pts'])
    rtest = get_rare_indices(dftestrare[target_variable], yreltest, thresh, phi_params['control.pts'])

    if len(rtrain) != numraretrain:
        raise ValueError('Incompatibility between the number of rare values that must be included in the '
                         'training data for equal class distribution and the obtained number of rare')

    if len(rtest) != numraretest:
        raise ValueError('Incompatibility between the number of rare values that must be included in the '
                         'testing data for equal class distribution and the obtained number of rare')

    # return dftrainrare, dftestrare, phi_params['control.pts']
    return dftrainrare, dftestrare, rtrain, rtest, yreltrain, yreltest, phi_params, loss_params, demandrel


def get_relevance_oversampling(smogned, target_variable, targetrel):
    '''
    gets the relevance values of an oversampled data frame
    :param smogned: the oversampled data frame
    :param target_variable: name of the target variable column
    :param targetrel: dictionary mapping each target variable value to a relevance value
    :return: the relevance of the oversampled data frame
    '''

    yrelafter = []
    distances = []
    for val in smogned[target_variable]:
        if val in targetrel:
            yrelafter.append(targetrel[val])
        else:
            nearest = min(sorted(list(targetrel.keys())), key=lambda x: abs(x - val))
            distances.append(abs(nearest - val))
            yrelafter.append(targetrel[nearest])

    return yrelafter, distances


class LearningModel:
    def __init__(self, df, target_variable, split_ratio: float,
                 output_folder,

                 # parameters for the phi.control :) :)
                 rel_method='extremes', extr_type='high', coef=1.5, relevance_pts=None,

                 # parameters for the SmoteRegress, RandUnder, GaussNoiseRegress, DIBSRegress
                 rel="auto", thr_rel=0.5, Cperc="balance",
                 k=5, repl=False, dist="Euclidean", p=2, pert=0.1,

                 # the over/under sampling method
                 smogn=True, rand_under=False, smoter=False, gn=False, nosmote=False,

                 # scaling input and output
                 cols_drop=None, scale=True, scale_input=True, scale_output=False,
                 output_zscore=False, output_minmax=False, output_box=False, output_log=False,
                 input_zscore=None, input_minmax=None, input_box=None, input_log=None,

                 service_name=None, mohafaza=None, testing_data=None,

                 grid=True, random_grid=False,
                 nb_folds_grid=None, nb_repeats_grid=None,

                 save_errors_xlsx=True,
                 save_best_testing_instances=False,
                 save_validation=False):
        '''
        #################################
        Parameters for the phi function
        :param rel_method: 'extremes' or 'range'
        :param extr_type: 'high', 'low', or 'both' (defualt)
        :param coef: 1.5
        :param relevance_pts: relevance matrix
        ################################

        #################################
        Parameters for the SMOGN, SMOTER, GN, AND DIBSRegress
        :param rel: 'auto' OR the relevance matrix
        :param thr_rel: the threshold
        :param Cperc: 'balance', 'extreme', or ascending order of target variable (make sure same nb as the
        nb of rows in the relevance matrix)
        :param k: k nearest neighbos
        :param repl:
        :param dist:
        :param p:
        :param pert
        :param smogn: False or True
        :param rand_under: False or True
        :param smoter: False or True
        :param gn: False or True
        ##################################
        '''

        if [smogn, smoter, rand_under, gn, nosmote].count(True) > 1:
            raise ValueError('Cannot have several methods set to True [smogn, rand_under, smoter, gn]')

        if [smogn, smoter, rand_under, gn, nosmote].count(False) == 5:
            raise ValueError('Cannot have all methods set to False [smogn, rand_under, smoter, gn]')

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

        # save a copy of the name of the target variable before it gets renamed (due to R issues)
        self.target_variable_old = target_variable

        # list of columns to drop
        self.cols_drop = cols_drop
        if self.cols_drop is not None:
            self.df_train = self.df_train.drop(self.cols_drop, axis=1)
            self.df_test = self.df_test.drop(self.cols_drop, axis=1)
            print('list of columns used in modeling')
            print(list(self.df_test.columns.values))

        # rename columns for R's formula
        for col in list(self.df_train.columns.values):
            newname = col
            for char in ['+', '-', '{', '}', ' ', '(', ')']:
                if char in col:
                    newname = newname.replace(char, '_')
            # self.data_train = self.data_train.rename(columns={col: newname})
            # self.data_test = self.data_test.rename(columns={col: newname})
            if newname != col:
                print('Renamed %s to %s' % (col, newname))
                self.df_train = self.df_train.rename(columns={col: newname})
                self.df_test = self.df_test.rename(columns={col: newname})
                self.df_test_orig = self.df_test_orig.rename(columns={col: newname})

                if col == self.target_variable:
                    print('target variable is renamed ...')
                    self.target_variable = newname

        # list of all column names except that of the target variable (used for creating R's formula)
        self.other = list(self.df_train.columns.values)
        self.other.remove(self.target_variable)

        # print('shuffling the 80% training before cv ...')
        # self.df_train = self.df_train.sample(frac=1, random_state=42)

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

        # the over/under sampling method used
        self.smogn = smogn
        self.rand_under = rand_under
        self.smoter = smoter
        self.gn = gn
        self.nosmote = nosmote

        print('\nsmogn: {}'.format(self.smogn))
        print('smoter: {}'.format(self.smoter))
        print('randUnder: {}'.format(self.rand_under))
        print('gn: {}'.format(self.gn))
        print('no_smote: {}\n'.format(self.nosmote))

        # parameters related to phi function (for relevance)

        # check that if rel_method == range, the user must have provided a relevance matrix (relevance_pts)
        if rel_method == 'range' and relevance_pts is None:
            raise ValueError('You have set rel_method = range. You must provide relevance_pts as a matrix.'
                             'Currently, it is None')

        self.rel_method = rel_method,
        self.extr_type = extr_type
        self.coef = coef
        self.relevance_pts = relevance_pts,

        # parameters related to Over/Under - Sampling
        self.rel = rel
        self.thr_rel = thr_rel
        self.Cperc = Cperc

        print('\nrel to used in case of SmoteR, GN, randUNder: {}'.format(self.rel))
        print('Threshold: {}'.format(self.thr_rel))
        print('Cperc: {}\n'.format(self.Cperc))

        self.k = k
        self.repl = repl
        self.dist = dist
        self.p = p
        self.pert = pert

        # related to cross validation
        self.grid = grid
        self.random_grid = random_grid
        self.nb_folds_grid = nb_folds_grid
        self.nb_repeats_grid = nb_repeats_grid

        self.split_ratio = split_ratio

        # df_train, df_test with equal class distribution between rare and not rare
        # dftrainrare: the training data that has X% rare values
        # dftestrare: the testing data that has X% rare values
        # rtrain: indices of rare values in dftrainrare
        # rtest: indices of rare values in dftestrare
        # yreltrain: the relevance values of dftrainrare
        # yreltest: the relevance values of dftestrare
        # phi_params: parameters of the phi function (from phi.control)
        # loss_params: parameters of the phi function (from phi.control)
        # targetrel: dictionary mapping each target variable value to a relevance value

        dftrainrare, dftestrare, rtrain, rtest, yreltrain, yreltest, phi_params, loss_params, targetrel = \
            shuffle_rarify_data(self.df_train, self.df_test,
                                self.target_variable, method=self.rel_method[0],
                                extr_type=self.extr_type, thresh=self.thr_rel,
                                coef=self.coef, control_pts=self.relevance_pts[0])

        # set self.df_train and self.df_test to the new shuffled data that has equal class
        # distribution between rare and not rare
        self.df_train = dftrainrare
        self.df_test = dftestrare

        self.rtrain = rtrain
        self.rtest = rtest
        self.yreltrain = yreltrain
        self.yreltest = yreltest
        self.phi_params = phi_params
        self.loss_params = loss_params
        self.targetrel = targetrel

        # save shuffled and rarified data to folder
        self.save_shuffled_dataset(dftrainrare, dftestrare)

        # save the rare indices of the shuffled data - Before Oversampling
        self.save_rare_before_oversampling(rtrain, rtest)

        # plot rare values in the training data before SMOGN/SMOTER/GN/RandUnderRegress/
        print('\n ********************* Rare Values - Before SMOTE ********************* ')
        plot_rare(self.df_train[self.target_variable], self.rtrain, self.target_variable,
                  output_folder=self.output_folder + 'plots/',
                  fig_name='rare_values_before_%s' % get_method(self.smogn, self.smoter, self.rand_under, self.gn))

        # plot the relevance of the target variable
        plot_relevance(self.df_train[self.target_variable], self.yreltrain, self.target_variable,
                       output_folder=self.output_folder + 'plots/',
                       fig_name='relevance_plot_%s' % get_method(self.smogn, self.smoter, self.rand_under, self.gn))

        # show plot of target variable before oversampling
        plot_target_variable(self.df_train, self.target_variable, self.output_folder + 'plots/',
                             '%s_not_oversampled' % (self.target_variable))

        # save error metrics to xlxs sheet
        self.save_errors_xlsx = save_errors_xlsx
        self.save_validation = save_validation
        self.save_best_testing_instances = save_best_testing_instances

        # save statistics about rare values in the data (before and after SMOGN) as a csv file
        self.rare_statistics = pd.DataFrame(columns=['total_rare_before', 'rare_train_before', 'rare_test_before',
                                                     'train_size_before', 'train_size_after', 'rare_train_after',
                                                     'Cperc', 'avg_dist'])

        # save the error metrics in a dataframe and save it later as csv file
        if self.save_errors_xlsx:
            self.results = pd.DataFrame(columns=['r2', 'adj-r2', 'rmse', 'mse', 'mae', 'mape',
                                                 'F1', 'F2', 'F05', 'prec', 'rec',
                                                 'nll', 'loss',
                                                 'avg_%s' % self.target_variable,
                                                 'winning_hyperparams',
                                                 'pearson', 'spearman', 'distance',
                                                 'train_time_min',
                                                 'train_time_sec'])

        else:
            self.results = None

        # service_name & mohafaza for MoPH
        self.service_name = service_name
        self.mohafaza = mohafaza

        # numpy arrays X_train, y_train, X_test, y_test

        df_without_target = self.df_train
        df_without_target = df_without_target.drop([self.target_variable], axis=1)
        self.feature_names = list(df_without_target.columns.values)
        print(self.feature_names)
        self.feat_representations = None

        # get the names of the columns of the one hot encodings
        self.one_hot_encoded = None
        if self.service_name is None and self.mohafaza is None:
            if 'service_General_Medicine' in list(self.df_train.columns.values):
                # applicable only if we are using the train_collated_test_collated dataset
                # because the separated datasets don't have these
                self.one_hot_encoded = self.feature_names[-7:]

        self.X_train = np.array(self.df_train.loc[:, self.df_train.columns != self.target_variable])
        self.y_train = np.array(self.df_train.loc[:, self.target_variable])

        self.X_test = np.array(self.df_test.loc[:, self.df_test.columns != self.target_variable])
        self.y_test = np.array(self.df_test.loc[:, self.target_variable])

        # density plot before applying any smote
        plot_density(self.df_train, self.target_variable, self.output_folder + 'plots/',
                     'density_before_%s' % get_method(self.smogn, self.smoter, self.rand_under, self.gn),
                     'Density Plot', None)

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

        # dictionary containing uncertainty scores
        self.uncertainties = {}

    def save_train_test_before_modeling(self):
        ''' save the training and testing data frames before any processing happens to them '''
        path = self.output_folder + 'train_test_before_modeling/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.df_train.to_csv(path + 'training.csv', index=False)
        self.df_test.to_csv(path + 'testing.csv', index=False)

    def inverse_boxcox(self, y_box, lambda_):
        pred_y = np.power((y_box * lambda_) + 1, 1 / lambda_) - 1
        return pred_y

    def count_abnormal(self, df):
        '''
        Due to Oversampling, SMOGN is adding noise to the one hot encoded vectors. This function counts how many of these
        are being done
        :param df: the oversampled data frame
        :return: statistics about the above
        '''
        count = 0
        for i, row in df.iterrows():
            if row['service_General_Medicine'] not in [0, 1]:
                count += 1
            elif row['service_Gynaecology'] not in [0, 1]:
                count += 1
            elif row['service_Pediatrics'] not in [0, 1]:
                count += 1
            elif row['service_Pharmacy'] not in [0, 1]:
                count += 1
            elif row['mohafaza_NE'] not in [0, 1]:
                count += 1
            elif row['mohafaza_B'] not in [0, 1]:
                count += 1
            elif row['mohafaza_N'] not in [0, 1]:
                count += 1
            else:
                continue

        print('number of noisy one hot encoded: {} out of {}'.format(count, len(df)))
        print('percentage of noisy one hot encoded: %.3f' % (count / len(df) * 100))

    def round_oversampled_one_hot_encoded(self, df):
        '''
        round one hot encoded vectors of an oversampled dataset. We have fed the SMOGN/SMOTER/GN/RandUnder
        a data frame having one hot encoded values (0s and 1s). However, given that we are using Euclidean/Manhattan
        distances for oversampling, some noise is added to these making them 1.0003, 0.99, etc.
        Having this said, this function will round these values back again so they are
        perfect 0s or 1s. We could have used HEOM distance, but it expects "nominal" features
        as opposed to one hot encodings.
        :param df: the over-sampled data frame
        :return: the over-sampled data frame with one hot encodings rounded
        '''
        for col in self.one_hot_encoded:
            df.loc[df[col] < 0.5, col] = 0
            df.loc[df[col] >= 0.5, col] = 1
        return df

    def save_shuffled_dataset(self, train_shuffled, test_shuffled):
        ''' save shuffled datasets and their rare indices - Before Oversampling
        :param train_shuffled: shuffled-rarified training data
        :param test_shuffled: shuffled-rarified testing data
        '''
        destination = self.output_folder + 'shuffled_data/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        train_shuffled.to_csv(destination + 'df_train_collated_shuffled.csv', index=False)
        test_shuffled.to_csv(destination + 'df_test_collated_shuffled.csv', index=False)

    def save_oversampled_dataset(self, oversampled, model_name):
        ''' saves the over sampled dataset into its own folder, might use for later analysis - After Oversampling
        :param oversampled: the oversampled data frame
        :param model_name: name of the model in use
        :return saves it in appropriate output folder: /output_folder/oversample_../
        '''
        destination = self.output_folder + 'oversampled_dataset/'
        if not os.path.exists(destination):
            os.makedirs(destination)
        # assuming we are oversampling only the training dataset
        oversampled.to_csv(destination + 'df_train_%s_%s.csv' % (
        model_name, get_method(self.smogn, self.smoter, self.rand_under, self.gn)), index=False)

    def save_rare_before_oversampling(self, rare_train, rare_test):
        '''
        :param rare_train: list of the rare indices of the shuffled training data
        :param rare_test: list of the rare indices of the shuffled testing data
        :return: dumps both lists into txt files using pickle
        '''

        destination = self.output_folder + 'rare_before_oversampling/'
        if not os.path.exists(destination):
            os.makedirs(destination)

        with open(destination + 'rare_indices_train.txt', "wb") as fp:
            pickle.dump(rare_train, fp)

        with open(destination + 'rare_indices_test.txt', "wb") as fp:
            pickle.dump(rare_test, fp)

    def save_rare_after_oversampling(self, rareindices, model_name):
        '''
        Saves the indices of rare values in the oversmapled <<training>> dataset
        :param rareindices: list of rare indices in the oversampled <<training>> dataset
        :param model_name: name of the current macho#ine learning model in use
        :return: dumps rare indices into a txt file using pickle
        '''

        destination = self.output_folder + 'rare_after_oversampling/'
        if not os.path.exists(destination):
            os.makedirs(destination)

        with open(destination + 'rare_indices_%s.txt' % model_name, "wb") as fp:
            pickle.dump(rareindices, fp)

    def apply_smogn(self, df_combined, plotdensity=False, model_name=None):
        '''
        method that applies SMOGN Algorithm to the current data frame
        '''
        if self.smogn:
            smogned = runit.WFDIBS(
                fmla=get_formula(self.target_variable),
                dat=pandas2ri.py2ri(df_combined),
                # dat=df,
                method=self.phi_params['method'][0],
                npts=self.phi_params['npts'][0],
                controlpts=self.phi_params['control.pts'],
                thrrel=self.thr_rel,
                Cperc=self.Cperc,
                k=self.k,
                repl=self.repl,
                dist=self.dist,
                p=self.p,
                pert=self.pert)

            smogned = pandas2ri.ri2py_dataframe(smogned)

            # shuffle the data once again after oversampling
            smogned = smogned.sample(frac=1, random_state=1).reset_index(drop=True)

            if plotdensity:
                # density plot on the oversampled data
                plot_density(smogned, self.target_variable, self.output_folder + 'plots/', 'density_after_smogn',
                             'Density Plot', model_name)

            X_train = np.array(smogned.loc[:, smogned.columns != self.target_variable])
            y_train = np.array(smogned.loc[:, self.target_variable])

            return smogned, X_train, y_train

    def apply_smoter(self, df_combined, plotdensity=False, model_name=None):

        '''
        method that applies SmoteRegress to the current data frame
        '''

        smotered = runit.WFSMOTE(
            fmla=get_formula(self.target_variable),
            train=pandas2ri.py2ri(df_combined),
            rel=self.rel,
            thrrel=self.thr_rel,
            Cperc=self.Cperc,
            k=self.k,
            repl=self.repl,
            dist=self.dist,
            p=self.p,
        )

        smotered = pandas2ri.ri2py_dataframe(smotered)

        # shuffle the data once again after oversampling
        smotered = smotered.sample(frac=1, random_state=1).reset_index(drop=True)

        if plotdensity:
            # density plot after oversampling
            plot_density(smotered, self.target_variable, self.output_folder + 'plots/', 'density_after_smoter',
                         'Density Plot', model_name)

        X_train = np.array(smotered.loc[:, smotered.columns != self.target_variable])
        y_train = np.array(smotered.loc[:, self.target_variable])


        return smotered, X_train, y_train

    def apply_gn(self, df_combined, plotdensity=False, model_name=None):

        gaussnoised = runit.WFGN(
            fmla=get_formula(self.target_variable),
            train=pandas2ri.py2ri(df_combined),
            rel=self.rel,
            thrrel=self.thr_rel,
            Cperc=self.Cperc,
            pert=self.pert,
            repl=self.repl
        )

        gaussnoised = pandas2ri.ri2py_dataframe(gaussnoised)

        # shuffle the data once again after oversampling
        gaussnoised = gaussnoised.sample(frac=1, random_state=1).reset_index(drop=True)

        if plotdensity:
            # density plot after smoting
            plot_density(gaussnoised, self.target_variable, self.output_folder + 'plots/', 'density_after_gn',
                         'Density Plot', model_name)

        X_train = np.array(gaussnoised.loc[:, gaussnoised.columns != self.target_variable])
        y_train = np.array(gaussnoised.loc[:, self.target_variable])

        return gaussnoised, X_train, y_train

    def apply_rand_under(self, df_combined, plotdensity=False, model_name=None):

        randunder = runit.WFRandUnder(
            fmla=get_formula(self.target_variable),
            train=pandas2ri.py2ri(df_combined),
            rel=self.rel,
            thrrel=self.thr_rel,
            Cperc=self.Cperc,
            repl=self.repl
        )

        randunder = pandas2ri.ri2py_dataframe(randunder)

        # shuffle the data once again after oversampling
        randunder = randunder.sample(frac=1, random_state=1).reset_index(drop=True)

        if plotdensity:
            # density plot after oversampling
            plot_density(randunder, self.target_variable, self.output_folder + 'plots/', 'density_after_randunder',
                         'Density Plot', model_name)

        X_train = np.array(randunder.loc[:, randunder.columns != self.target_variable])
        y_train = np.array(randunder.loc[:, self.target_variable])

        return randunder, X_train, y_train

    def prepare_data(self, X_train, y_train):
        '''
        concatenates X_train and y_train into one, and make them a data frame
        so we are able to process the data frame by SMOGN, RandUnder, GN, or SMOTER
        '''

        # reshape + rename
        X_train_samp = X_train
        y_train_samp = y_train.reshape(-1, 1)

        # combine two numpy arrays together into one numpy array
        combined = np.concatenate((X_train_samp, y_train_samp), axis=1)

        column_names = self.other + [self.target_variable]
        df_combined = pd.DataFrame(combined, columns=column_names)

        return df_combined

    def get_cmap(self, n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def plot_specific_probas(self, ngboost_tr, X_test, y_test, cands1, cands2,
                            cands1_lab, cands2_lab,
                            output_folder):
        '''

        :param ngboost_tr: trained ngboost model
        :param X_test: testing input data
        :param y_test: testing actual output data
        :param cands1: candidate point in testing
        :param cands2: candidate point in testing
        :param output_folder: folder to save plot in
        :param fig_name: name of the figure to be saved as such
        :return:
        '''

        c = np.arange(1, len(cands1) + 1)
        norm = matplotlib.colors.Normalize(vmin=c.min(), vmax=c.max())
        cmap_blue = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Blues)
        cmap_blue.set_array([])

        cmap_red = matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.cm.Reds)
        cmap_red.set_array([])

        cmaps = [cmap_blue, cmap_red]
        cands_sets = [cands1, cands2]
        labs = [cands1_lab, cands2_lab]

        for j, cands in enumerate(cands_sets):
            for i, cand in enumerate(cands):
                y_pred_dist = ngboost_tr.pred_dist(X_test[cand, :].reshape(1, -1))
                x_span = np.linspace(min(y_test), max(y_test), 100)
                dist_values = y_pred_dist.pdf(x_span)
                if i == len(cands) - 1:
                    plt.plot(x_span, dist_values, c=cmaps[j].to_rgba(i + 1))
                else:
                    plt.plot(x_span, dist_values, c=cmaps[j].to_rgba(i + 1))

            plt.legend(loc="upper right")
            plt.xlabel('{} values'.format(self.target_variable_old))
            plt.ylabel('probability')

            self.mkdir(output_folder)

            plt.savefig(os.path.join(output_folder, labs[j]))
            plt.close()

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

    def cross_validation_uncertainties(self, possible_hyperparams, model_name):

        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        # training and testing data if we have single target variable
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        if not self.nosmote:
            print('\nSize of the data before Oversampling: %d\n' % len(y_train))

        self.len_train_before = len(y_train)

        # indices of the rare values
        rare_values = self.rtrain
        y_train_old = y_train
        rare_vec = [1 if i in rare_values else 0 for i in range(len(y_train))]

        y_train = np.array(rare_vec)

        tempModels = []

        # specify the type of cv (kfold vs. repeated kfold)
        if self.nb_repeats_grid is None:
            print('running Stratified %d-fold cross validation' % self.nb_folds_grid)
            kf = StratifiedKFold(n_splits=self.nb_folds_grid)
        else:
            print('running Repeated Stratified %d-fold-%d-repeats cross validation' % (
                self.nb_folds_grid, self.nb_repeats_grid))
            kf = RepeatedStratifiedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid,
                                         random_state=2652124)
        parameters = possible_hyperparams

        # hyper parameters loop
        t1 = time.time()
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []
            f1_scores, f2_scores, f05_scores, prec_scores, rec_scores = [], [], [], [], []
            epis_uncertainties, ale_uncertainties, tot_uncertainties = [], [], []

            count = 0
            for train_index, test_index in kf.split(X_train, y_train):
                tf.reset_default_graph()

                X_train_inner, X_val = X_train[train_index], X_train[test_index]

                # replace the y's with their original values after splitting in a stratified manner
                y_train_inner, y_val = y_train_old[train_index], y_train_old[test_index]

                drp = parameter['dropout']
                lr = parameter['learning_rate']
                epochs = parameter['epochs']

                if self.scale:
                    X_train_inner, X_val, y_train_inner, y_val, scaler_out_final = self.scale_cols(X_train_inner, X_val,
                                                                                                   y_train_inner, y_val)

                if model_name == 'bootstrap':
                    n_heads = parameter['n_heads']
                    # does not differ between epistemic and aleatoric uncertainty
                    y_pred, uncertainty_v, loss = bootstrap_evaluation(X_train=X_train_inner, y_train=y_train_inner,
                                                                       X_test=X_val, y_test=y_val,
                                                                       dropout=drp, learning_rate=lr, epochs=epochs,
                                                                       n_heads=n_heads)

                    tot_uncertainties.append(np.mean(uncertainty_v))

                elif model_name == 'combined':
                    n_passes = parameter['n_passes']
                    y_pred, epistemic_unc_v, aleatoric_unc_v, total_unc_v, loss = combined_evaluation(
                        X_train=X_train_inner, y_train=y_train_inner, X_test=X_val, y_test=y_val, dropout=drp, learning_rate=lr, epochs=epochs,
                        n_passes=n_passes)
                    epis_uncertainties.append(np.mean(epistemic_unc_v))
                    ale_uncertainties.append(np.mean(aleatoric_unc_v))
                    tot_uncertainties.append(np.mean(total_unc_v))
                else:
                    # the mixture
                    n_mixtures = parameter['n_mixtures']
                    y_pred, epistemic_unc_v, aleatoric_unc_v, total_unc_v, loss = \
                        mixture_evaluation(X_train=X_train_inner, y_train=y_train_inner, X_test=X_val,
                                           y_test=y_val, dropout=drp, learning_rate=lr, epochs=epochs,
                                           n_mixtures=n_mixtures)

                    epis_uncertainties.append(np.mean(epistemic_unc_v))
                    ale_uncertainties.append(np.mean(aleatoric_unc_v))
                    tot_uncertainties.append(np.mean(total_unc_v))

                y_pred = y_pred.flatten() if y_pred.ndim == 2 else y_pred

                r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance, rmetrics = get_stats(y_val, y_pred,
                                                                                                    X_val.shape[1],
                                                                                                    self.thr_rel,
                                                                                                    self.phi_params,
                                                                                                    self.loss_params)

                # add point predictions score
                r2_scores.append(r2)
                adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
                f1_scores.append(rmetrics['ubaF1'][0])
                f2_scores.append(rmetrics['ubaF2'][0])
                f05_scores.append(rmetrics['ubaF05'][0])
                prec_scores.append(rmetrics['ubaprec'][0])
                rec_scores.append(rmetrics['ubarec'][0])

            tempModels.append(
                [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores), np.mean(mse_scores),
                 np.mean(mae_scores), np.mean(mape_scores),
                 np.mean(f1_scores), np.mean(f2_scores), np.mean(f05_scores),
                 np.mean(prec_scores), np.mean(rec_scores),
                 np.mean(epis_uncertainties), np.mean(ale_uncertainties), np.mean(tot_uncertainties)])

            count += 1

        tf.reset_default_graph()

        tempModels = sorted(tempModels, key=lambda k: k[3])
        winning_hyperparameters = tempModels[0][0]

        print('winning hyper parameters: ', str(winning_hyperparameters))
        print('Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
              'F1: %.5f\nF2: %.5f\nF0.5: %.5f\nprec: %.5f\nrec: %.5f\n' %
              (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
               tempModels[0][6],
               tempModels[0][7], tempModels[0][8], tempModels[0][9], tempModels[0][10], tempModels[0][11]))

        if self.save_errors_xlsx:
            if self.save_validation:
                self.results.loc['%s_val' % model_name] = pd.Series({
                    'r2': tempModels[0][1],
                    'adj-r2': tempModels[0][2],
                    'rmse': tempModels[0][3],
                    'mse': tempModels[0][4],
                    'mae': tempModels[0][5],
                    'mape': tempModels[0][6],
                    'F1': tempModels[0][7],
                    'F2': tempModels[0][8],
                    'F05': tempModels[0][9],
                    'prec': tempModels[0][10],
                    'rec': tempModels[0][11],
                })

        if model_name == 'bootstrap':
            self.uncertainties[model_name] = {
                'val_epistemic_unc': 0,
                'val_aleatoric_unc': 0,
                'val_total_unc': tempModels[0][14]
            }
        else:
            self.uncertainties[model_name] = {
                'val_epistemic_unc': tempModels[0][12],
                'val_aleatoric_unc': tempModels[0][13],
                'val_total_unc': tempModels[0][14]
            }

        # replace back y_train to the old one after finishing stratified k fold cross validation
        y_train = y_train_old

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        if not self.nosmote:
            # apply over/under sampling only on training data

            df_combined = self.prepare_data(X_train, y_train)

            # transform one hot encodings from float to int
            if self.one_hot_encoded is not None:
                df_combined = convert_cols_int(df_combined, self.one_hot_encoded)

            if self.smogn:
                oversampled, X_train, y_train = self.apply_smogn(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            elif self.smoter:
                oversampled, X_train, y_train = self.apply_smoter(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            elif self.gn:
                oversampled, X_train, y_train = self.apply_gn(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            else:
                oversampled, X_train, y_train = self.apply_rand_under(df_combined, plotdensity=True,
                                                                      model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            # show plot of target variable after oversampling
            plot_target_variable(oversampled, self.target_variable, self.output_folder + 'plots/',
                                 '%s_oversampled_%s_%s' % (
                                     self.target_variable, get_method(self.smogn, self.smoter, self.rand_under,
                                                                      self.gn), model_name))

            print('\n ********************* Rare Values - After SMOTE ********************* ')
            yrelafter, distances = get_relevance_oversampling(oversampled, self.target_variable, self.targetrel)
            roversampled = get_rare_indices(oversampled[self.target_variable], yrelafter, self.thr_rel,
                                            self.phi_params['control.pts'])

            # save rare indices of the oversampled <<training>> data - i.e. After Oversampling
            self.save_rare_after_oversampling(roversampled, model_name)

            avg_dist = np.mean(distances)

            self.rare_statistics.loc[model_name] = pd.Series({
                'total_rare_before': (len(self.rtrain + self.rtest) / (len(self.df_train) + len(self.df_test))) * 100,
                'rare_train_before': (len(self.rtrain) / len(self.df_train)) * 100,
                'rare_test_before': (len(self.rtest) / len(self.df_test)) * 100,
                'train_size_before': self.len_train_before,
                'train_size_after': len(oversampled),
                'rare_train_after': (len(roversampled) / len(oversampled)) * 100,
                'Cperc': self.Cperc,
                'avg_dist': avg_dist
            })

            # plots the rare indices
            plot_rare(oversampled[self.target_variable], roversampled, self.target_variable,
                      output_folder=self.output_folder + 'plots/',
                      fig_name='rare_values_after_%s' % get_method(self.smogn, self.smoter, self.rand_under, self.gn),
                      model_name=model_name)

            print('Average distances between actual and synthesized: {}'.format(avg_dist))

        drp = winning_hyperparameters['dropout']
        lr = winning_hyperparameters['learning_rate']
        epochs = winning_hyperparameters['epochs']

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
                                                             X_test=X_test, y_test=y_test, dropout=drp,
                                                            learning_rate=lr,epochs=epochs, n_passes=n_passes)

        else:
            # the mixture
            n_mixtures = winning_hyperparameters['n_mixtures']
            y_pred, epistemic_unc, aleatoric_unc, total_unc, loss = mixture_evaluation(X_train=X_train, y_train=y_train,
                                                             X_test=X_test, y_test=y_test, dropout=drp,
                                                             learning_rate=lr, epochs=epochs,
                                                             n_mixtures=n_mixtures)

        t2 = time.time()
        time_taken_min = float(t2 - t1) / 60
        time_taken_sec = float(t2 - t1)

        self.plot_actual_vs_predicted(y_test, y_pred,
                                      self.output_folder + 'train_test_forecasts_lineplot/',
                                      model_name)

        self.plot_actual_vs_predicted_scatter_bisector(y_test, y_pred,
                                                       self.output_folder + 'train_test_forecasts_scatterplot_bisector/',
                                                       model_name)

        y_pred = y_pred.flatten() if y_pred.ndim == 2 else y_pred

        r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance, rmetrics = get_stats(y_test, y_pred,
                                                                                            X_test.shape[1],
                                                                                            self.thr_rel,
                                                                                            self.phi_params,
                                                                                            self.loss_params)
        print('\nTesting Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
              'F1: %.5f\nF2: %.5f\nF0.5: %.5f\nprec: %.5f\nrec: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape,
               rmetrics['ubaF1'][0], rmetrics['ubaF2'][0], rmetrics['ubaF05'][0], rmetrics['ubaprec'][0],
               rmetrics['ubarec'][0]))

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))
        print('Pearson Correlation: %.5f' % pearson)
        print('Spearman Correlation: %.5f' % spearman)
        print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        if self.save_errors_xlsx:
            if self.save_validation:
                row_name = '%s_test' % model_name
            else:
                row_name = model_name

            self.results.loc[row_name] = pd.Series({'r2': r2, 'adj-r2': adj_r2, 'rmse': rmse, 'mse': mse,
                                                    'mae': mae, 'mape': mape,
                                                    'F1': rmetrics['ubaF1'][0],
                                                    'F2': rmetrics['ubaF2'][0],
                                                    'F05': rmetrics['ubaF05'][0],
                                                    'prec': rmetrics['ubaprec'][0],
                                                    'rec': rmetrics['ubarec'][0],
                                                    'avg_%s' % self.target_variable: avg_target,
                                                    'loss': loss,
                                                    'winning_hyperparams': str(winning_hyperparameters),
                                                    'pearson': pearson, 'spearman': spearman,
                                                    'distance': distance,
                                                    'train_time_min': time_taken_min,
                                                    'train_time_sec': time_taken_sec})

        if model_name == 'bootstrap':
            self.uncertainties[model_name]['test_epistemic_unc'] = 0
            self.uncertainties[model_name]['test_aleatoric_unc'] = 0
            self.uncertainties[model_name]['test_total_unc'] = np.mean(uncertainty)

        else:
            self.uncertainties[model_name]['test_epistemic_unc'] = np.mean(epistemic_unc)
            self.uncertainties[model_name]['test_aleatoric_unc'] = np.mean(aleatoric_unc)
            self.uncertainties[model_name]['test_total_unc'] = np.mean(total_unc)

    def cross_validation_grid_deep_ensemble(self, possible_hyperparams):
        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        model_name = 'deep_ensemble'

        # training and testing data if we have single target variable
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        if not self.nosmote:
            print('\nSize of the data before Oversampling: %d\n' % len(y_train))

        self.len_train_before = len(y_train)

        # indices of the rare values
        rare_values = self.rtrain
        y_train_old = y_train
        rare_vec = [1 if i in rare_values else 0 for i in range(len(y_train))]

        y_train = np.array(rare_vec)

        tempModels = []

        # specify the type of cv (kfold vs. repeated kfold)
        if self.nb_repeats_grid is None:
            print('running Stratified %d-fold cross validation' % self.nb_folds_grid)
            kf = StratifiedKFold(n_splits=self.nb_folds_grid)
        else:
            print('running Repeated Stratified %d-fold-%d-repeats cross validation' % (
                self.nb_folds_grid, self.nb_repeats_grid))
            kf = RepeatedStratifiedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid,
                                         random_state=2652124)

        t1 = time.time()

        parameters = possible_hyperparams

        # hyper parameters loop
        t1 = time.time()
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):

            de_nll = []
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []
            f1_scores, f2_scores, f05_scores, prec_scores, rec_scores = [], [], [], [], []
            uncertainties = []

            count = 0
            for train_index, test_index in kf.split(X_train, y_train):

                tf.reset_default_graph()

                X_train_inner, X_val = X_train[train_index], X_train[test_index]

                # replace the y's with their original values after splitting in a stratified manner
                y_train_inner, y_val = y_train_old[train_index], y_train_old[test_index]

                lr = parameter['learning_rate']
                bs = parameter['batch_size']
                epochs = parameter['max_iter']
                layers = parameter['sizes']
                opt = parameter['optimizer_name']

                if self.scale:
                    X_train_inner, X_val, y_train_inner, y_val, scaler_out_final = self.scale_cols(X_train_inner, X_val,
                                                                                                   y_train_inner, y_val)

                    # mean, var, nll, outputs_per_ensemble
                y_pred, var, de_nll_val, _ = deep_ensemble_training.train_ensemble(X_train_inner, y_train_inner, X_val, y_val, learning_rate=lr,
                                                         max_iter=epochs, batch_size=bs,
                                                         sizes=layers, optimizer_name=opt,
                                                         produce_plots=False)

                if len(y_pred.shape) == 2:
                    y_pred = y_pred.reshape(-1)
                if len(y_val.shape) == 1:
                    y_val = y_val.reshape(-1)

                r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance, rmetrics = get_stats(y_val, y_pred,
                                                                                                    X_val.shape[1],
                                                                                                    self.thr_rel,
                                                                                                    self.phi_params,
                                                                                                    self.loss_params)

                # add point predictions score
                r2_scores.append(r2)
                adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
                f1_scores.append(rmetrics['ubaF1'][0])
                f2_scores.append(rmetrics['ubaF2'][0])
                f05_scores.append(rmetrics['ubaF05'][0])
                prec_scores.append(rmetrics['ubaprec'][0])
                rec_scores.append(rmetrics['ubarec'][0])

                # add scores probabilistic predictions
                de_nll.append(de_nll_val)

                # add the uncertainties
                uncertainties.append(np.mean(var))

            tempModels.append(
                [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores), np.mean(mse_scores),
                 np.mean(mae_scores), np.mean(mape_scores),
                 np.mean(f1_scores), np.mean(f2_scores), np.mean(f05_scores),
                 np.mean(prec_scores), np.mean(rec_scores),
                 np.mean(de_nll), np.std(de_nll), np.mean(uncertainties)])

            count += 1

        # reset the graph one last time for the final training/testing
        tf.reset_default_graph()

        tempModels = sorted(tempModels, key=lambda k: k[3])
        winning_hyperparameters = tempModels[0][0]

        self.nlls['deep_ensemble'] = {
            'mean': tempModels[0][12],
            'std': tempModels[0][13]
        }

        self.uncertainties[model_name] = {
            'val_epistemic_unc': 0,
            'val_aleatoric_unc': 0,
            'val_total_unc': tempModels[0][14]
        }

        print('winning hyper parameters: ', str(winning_hyperparameters))
        print('Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
              'F1: %.5f\nF2: %.5f\nF0.5: %.5f\nprec: %.5f\nrec: %.5f\nDE-NLL: %.5f +/- %.5f\n' %
              (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
               tempModels[0][6],
               tempModels[0][7], tempModels[0][8], tempModels[0][9], tempModels[0][10], tempModels[0][11],
               tempModels[0][12], tempModels[0][13]))

        if self.save_errors_xlsx:
            if self.save_validation:
                self.results.loc['%s_val' % model_name] = pd.Series({
                    'r2': tempModels[0][1],
                    'adj-r2': tempModels[0][2],
                    'rmse': tempModels[0][3],
                    'mse': tempModels[0][4],
                    'mae': tempModels[0][5],
                    'mape': tempModels[0][6],
                    'F1': tempModels[0][7],
                    'F2': tempModels[0][8],
                    'F05': tempModels[0][9],
                    'prec': tempModels[0][10],
                    'rec': tempModels[0][11],
                    'nll': tempModels[0][12]
                })

        # replace back y_train to the old one after finishing stratified k fold cross validation
        y_train = y_train_old

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        if not self.nosmote:
            # apply over/under sampling only on training data

            df_combined = self.prepare_data(X_train, y_train)

            # transform one hot encodings from float to int
            if self.one_hot_encoded is not None:
                df_combined = convert_cols_int(df_combined, self.one_hot_encoded)

            if self.smogn:
                oversampled, X_train, y_train = self.apply_smogn(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            elif self.smoter:
                oversampled, X_train, y_train = self.apply_smoter(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            elif self.gn:
                oversampled, X_train, y_train = self.apply_gn(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            else:
                oversampled, X_train, y_train = self.apply_rand_under(df_combined, plotdensity=True,
                                                                      model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            # show plot of target variable after oversampling
            plot_target_variable(oversampled, self.target_variable, self.output_folder + 'plots/',
                                 '%s_oversampled_%s_%s' % (
                                     self.target_variable, get_method(self.smogn, self.smoter, self.rand_under,
                                                                      self.gn), model_name))

            print('\n ********************* Rare Values - After SMOTE ********************* ')
            yrelafter, distances = get_relevance_oversampling(oversampled, self.target_variable, self.targetrel)
            roversampled = get_rare_indices(oversampled[self.target_variable], yrelafter, self.thr_rel,
                                            self.phi_params['control.pts'])

            # save rare indices of the oversampled <<training>> data - i.e. After Oversampling
            self.save_rare_after_oversampling(roversampled, model_name)

            avg_dist = np.mean(distances)

            self.rare_statistics.loc[model_name] = pd.Series({
                'total_rare_before': (len(self.rtrain + self.rtest) / (len(self.df_train) + len(self.df_test))) * 100,
                'rare_train_before': (len(self.rtrain) / len(self.df_train)) * 100,
                'rare_test_before': (len(self.rtest) / len(self.df_test)) * 100,
                'train_size_before': self.len_train_before,
                'train_size_after': len(oversampled),
                'rare_train_after': (len(roversampled) / len(oversampled)) * 100,
                'Cperc': self.Cperc,
                'avg_dist': avg_dist
            })

            # plots the rare indices
            plot_rare(oversampled[self.target_variable], roversampled, self.target_variable,
                      output_folder=self.output_folder + 'plots/',
                      fig_name='rare_values_after_%s' % get_method(self.smogn, self.smoter, self.rand_under, self.gn),
                      model_name=model_name)

            print('Average distances between actual and synthesized: {}'.format(avg_dist))

        lr = winning_hyperparameters['learning_rate']
        bs = winning_hyperparameters['batch_size']
        epochs = winning_hyperparameters['max_iter']
        layers = winning_hyperparameters['sizes']
        opt = winning_hyperparameters['optimizer_name']

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        # mean, var, nll, outputs_per_ensemble
        y_pred, var, de_nll_test, _ = deep_ensemble_training.train_ensemble(X_train, y_train, X_test, y_test, learning_rate=lr,
                                                  max_iter=epochs, batch_size=bs,
                                                  sizes=layers, optimizer_name=opt,
                                                  produce_plots=True,
                                                  output_folder=os.path.join(self.output_folder,
                                                                             'probabilistic_forecasts/'),
                                                  fig_name='deep_ensemble')

        self.uncertainties[model_name]['test_epistemic_unc'] = 0
        self.uncertainties[model_name]['test_aleatoric_unc'] = 0
        self.uncertainties[model_name]['test_total_unc'] = np.mean(var)

        t2 = time.time()
        time_taken_min = float(t2 - t1) / 60
        time_taken_sec = float(t2 - t1)
        
        self.plot_actual_vs_predicted(y_test, y_pred,
                                      self.output_folder + 'train_test_forecasts_lineplot/',
                                      model_name)

        self.plot_actual_vs_predicted_scatter_bisector(y_test, y_pred,
                                                       self.output_folder + 'train_test_forecasts_scatterplot_bisector/',
                                                       model_name)

        if len(y_pred.shape) == 2:
            y_pred = y_pred.reshape(-1)
        if len(y_test.shape) == 1:
            y_test = y_test.reshape(-1)

        r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance, rmetrics = get_stats(y_test, y_pred,
                                                                                            X_test.shape[1],
                                                                                            self.thr_rel,
                                                                                            self.phi_params,
                                                                                            self.loss_params)
        print('\nTesting Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
              'F1: %.5f\nF2: %.5f\nF0.5: %.5f\nprec: %.5f\nrec: %.5f\nDE-NLL: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape,
               rmetrics['ubaF1'][0], rmetrics['ubaF2'][0], rmetrics['ubaF05'][0], rmetrics['ubaprec'][0],
               rmetrics['ubarec'][0], de_nll_test))

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))
        print('Pearson Correlation: %.5f' % pearson)
        print('Spearman Correlation: %.5f' % spearman)
        print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        if self.save_errors_xlsx:
            if self.save_validation:
                row_name = '%s_test' % model_name
            else:
                row_name = model_name

            self.results.loc[row_name] = pd.Series({'r2': r2, 'adj-r2': adj_r2, 'rmse': rmse, 'mse': mse,
                                                    'mae': mae, 'mape': mape,
                                                    'F1': rmetrics['ubaF1'][0],
                                                    'F2': rmetrics['ubaF2'][0],
                                                    'F05': rmetrics['ubaF05'][0],
                                                    'prec': rmetrics['ubaprec'][0],
                                                    'rec': rmetrics['ubarec'][0],
                                                    'nll': de_nll_test,
                                                    'winning_hyperparams': str(winning_hyperparameters),
                                                    'avg_%s' % self.target_variable: avg_target,
                                                    'pearson': pearson, 'spearman': spearman,
                                                    'distance': distance,
                                                    'train_time_min': time_taken_min,
                                                    'train_time_sec': time_taken_sec})
            
            self.nlls['deep_ensemble']['test'] = de_nll_test
            
    def cross_validation_grid_mc_dropout(self, possible_hyperparams):
        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        model_name = 'mc_dropout'

        # training and testing data if we have single target variable
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        if not self.nosmote:
            print('\nSize of the data before Oversampling: %d\n' % len(y_train))

        self.len_train_before = len(y_train)

        # indices of the rare values
        rare_values = self.rtrain
        y_train_old = y_train
        rare_vec = [1 if i in rare_values else 0 for i in range(len(y_train))]

        y_train = np.array(rare_vec)

        tempModels = []

        # specify the type of cv (kfold vs. repeated kfold)
        if self.nb_repeats_grid is None:
            print('running Stratified %d-fold cross validation' % self.nb_folds_grid)
            kf = StratifiedKFold(n_splits=self.nb_folds_grid)
        else:
            print('running Repeated Stratified %d-fold-%d-repeats cross validation' % (
                self.nb_folds_grid, self.nb_repeats_grid))
            kf = RepeatedStratifiedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid,
                                         random_state=2652124)

        t1 = time.time()

        parameters = possible_hyperparams

        # hyper parameters loop
        t1 = time.time()
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):

            mc_nll, uncertainties = [], []
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []
            f1_scores, f2_scores, f05_scores, prec_scores, rec_scores = [], [], [], [], []

            count = 0
            for train_index, test_index in kf.split(X_train, y_train):

                X_train_inner, X_val = X_train[train_index], X_train[test_index]

                # replace the y's with their original values after splitting in a stratified manner
                y_train_inner, y_val = y_train_old[train_index], y_train_old[test_index]

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

                r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance, rmetrics = get_stats(y_val, y_pred,
                                                                                                    X_val.shape[1],
                                                                                                    self.thr_rel,
                                                                                                    self.phi_params,
                                                                                                    self.loss_params)

                # add point predictions score
                r2_scores.append(r2)
                adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
                f1_scores.append(rmetrics['ubaF1'][0])
                f2_scores.append(rmetrics['ubaF2'][0])
                f05_scores.append(rmetrics['ubaF05'][0])
                prec_scores.append(rmetrics['ubaprec'][0])
                rec_scores.append(rmetrics['ubarec'][0])

                # add the MC dropout probabilistic predictions
                mc_nll.append(MC_nll)

                # add the uncertainty
                uncertainties.append(np.mean(uncertainty))

                count += 1

            tempModels.append(
                [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores), np.mean(mse_scores),
                 np.mean(mae_scores), np.mean(mape_scores),
                 np.mean(f1_scores), np.mean(f2_scores), np.mean(f05_scores),
                 np.mean(prec_scores), np.mean(rec_scores),
                 np.mean(mc_nll), np.std(mc_nll), np.mean(uncertainties)])

        tempModels = sorted(tempModels, key=lambda k: k[3])
        winning_hyperparameters = tempModels[0][0]
        
        self.nlls['mc_dropout'] = {
            'mean': tempModels[0][12],
            'std': tempModels[0][13],
        }

        self.uncertainties[model_name] = {
            'val_epistemic_unc': 0,
            'val_aleatoric_unc': 0,
            'val_total_unc': tempModels[0][14]
        }

        print('winning hyper parameters: ', str(winning_hyperparameters))
        print('Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
              'F1: %.5f\nF2: %.5f\nF0.5: %.5f\nprec: %.5f\nrec: %.5f\nMC-NLL: %.5f +/- %.5f\n' %
              (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
               tempModels[0][6],
               tempModels[0][7], tempModels[0][8], tempModels[0][9], tempModels[0][10], tempModels[0][11],
               tempModels[0][12], tempModels[0][13]))

        if self.save_errors_xlsx:
            if self.save_validation:
                self.results.loc['%s_val' % model_name] = pd.Series({
                    'r2': tempModels[0][1],
                    'adj-r2': tempModels[0][2],
                    'rmse': tempModels[0][3],
                    'mse': tempModels[0][4],
                    'mae': tempModels[0][5],
                    'mape': tempModels[0][6],
                    'F1': tempModels[0][7],
                    'F2': tempModels[0][8],
                    'F05': tempModels[0][9],
                    'prec': tempModels[0][10],
                    'rec': tempModels[0][11],
                    'nll': tempModels[0][12]
                })

        # replace back y_train to the old one after finishing stratified k fold cross validation
        y_train = y_train_old

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        if not self.nosmote:
            # apply over/under sampling only on training data

            df_combined = self.prepare_data(X_train, y_train)

            # transform one hot encodings from float to int
            if self.one_hot_encoded is not None:
                df_combined = convert_cols_int(df_combined, self.one_hot_encoded)

            if self.smogn:
                oversampled, X_train, y_train = self.apply_smogn(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            elif self.smoter:
                oversampled, X_train, y_train = self.apply_smoter(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            elif self.gn:
                oversampled, X_train, y_train = self.apply_gn(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            else:
                oversampled, X_train, y_train = self.apply_rand_under(df_combined, plotdensity=True,
                                                                      model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            # show plot of target variable after oversampling
            plot_target_variable(oversampled, self.target_variable, self.output_folder + 'plots/',
                                 '%s_oversampled_%s_%s' % (
                                     self.target_variable, get_method(self.smogn, self.smoter, self.rand_under,
                                                                      self.gn), model_name))

            print('\n ********************* Rare Values - After SMOTE ********************* ')
            yrelafter, distances = get_relevance_oversampling(oversampled, self.target_variable, self.targetrel)
            roversampled = get_rare_indices(oversampled[self.target_variable], yrelafter, self.thr_rel,
                                            self.phi_params['control.pts'])

            # save rare indices of the oversampled <<training>> data - i.e. After Oversampling
            self.save_rare_after_oversampling(roversampled, model_name)

            avg_dist = np.mean(distances)

            self.rare_statistics.loc[model_name] = pd.Series({
                'total_rare_before': (len(self.rtrain + self.rtest) / (len(self.df_train) + len(self.df_test))) * 100,
                'rare_train_before': (len(self.rtrain) / len(self.df_train)) * 100,
                'rare_test_before': (len(self.rtest) / len(self.df_test)) * 100,
                'train_size_before': self.len_train_before,
                'train_size_after': len(oversampled),
                'rare_train_after': (len(roversampled) / len(oversampled)) * 100,
                'Cperc': self.Cperc,
                'avg_dist': avg_dist
            })

            # plots the rare indices
            plot_rare(oversampled[self.target_variable], roversampled, self.target_variable,
                      output_folder=self.output_folder + 'plots/',
                      fig_name='rare_values_after_%s' % get_method(self.smogn, self.smoter, self.rand_under, self.gn),
                      model_name=model_name)

            print('Average distances between actual and synthesized: {}'.format(avg_dist))

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

        y_pred, yt_hat, MC_nll_test, uncertainty = model.predict(X_test, y_test)

        self.uncertainties[model_name]['test_epistemic_unc'] = 0
        self.uncertainties[model_name]['test_aleatoric_unc'] = 0
        self.uncertainties[model_name]['test_total_unc'] = np.mean(uncertainty)

        # save the model
        models_folder = self.output_folder + 'trained_models/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        pkl_filename = "%s.pkl" % model_name
        with open(models_folder + pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        print('saved model to {} as {}.pkl'.format(models_folder, model_name))

        if not self.nosmote:
            print('\nSize of the data after Oversampling: %d\n' % len(y_train))

        t2 = time.time()
        time_taken_min = float(t2 - t1) / 60
        time_taken_sec = float(t2 - t1)

        output_dataset = self.create_output_dataset(y_pred, model_name, self.output_folder + 'output_vector_datasets/')

        self.plot_actual_vs_predicted(y_test, y_pred,
                                      self.output_folder + 'train_test_forecasts_lineplot/',
                                      model_name)

        self.plot_actual_vs_predicted_scatter_bisector(y_test, y_pred,
                                                       self.output_folder + 'train_test_forecasts_scatterplot_bisector/',
                                                       model_name)

        r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance, rmetrics = get_stats(y_test, y_pred,
                                                                                            X_test.shape[1],
                                                                                            self.thr_rel,
                                                                                            self.phi_params,
                                                                                            self.loss_params)
        print('\nTesting Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
              'F1: %.5f\nF2: %.5f\nF0.5: %.5f\nprec: %.5f\nrec: %.5f\nMC-NLL: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape,
               rmetrics['ubaF1'][0], rmetrics['ubaF2'][0], rmetrics['ubaF05'][0], rmetrics['ubaprec'][0],
               rmetrics['ubarec'][0], MC_nll_test))

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))
        print('Pearson Correlation: %.5f' % pearson)
        print('Spearman Correlation: %.5f' % spearman)
        print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        if self.save_errors_xlsx:
            if self.save_validation:
                row_name = '%s_test' % model_name
            else:
                row_name = model_name

            self.results.loc[row_name] = pd.Series({'r2': r2, 'adj-r2': adj_r2, 'rmse': rmse, 'mse': mse,
                                                    'mae': mae, 'mape': mape,
                                                    'F1': rmetrics['ubaF1'][0],
                                                    'F2': rmetrics['ubaF2'][0],
                                                    'F05': rmetrics['ubaF05'][0],
                                                    'prec': rmetrics['ubaprec'][0],
                                                    'rec': rmetrics['ubarec'][0],
                                                    'nll': MC_nll_test,
                                                    'winning_hyperparams': str(winning_hyperparameters),
                                                    'avg_%s' % self.target_variable: avg_target,
                                                    'pearson': pearson, 'spearman': spearman,
                                                    'distance': distance,
                                                    'train_time_min': time_taken_min,
                                                    'train_time_sec': time_taken_sec})
        
        self.nlls['mc_dropout']['test'] = MC_nll_test
        
        if not os.path.exists(self.output_folder + '/winning_hyperparams/'):
            os.makedirs(self.output_folder + '/winning_hyperparams/')
        with open(self.output_folder + 'winning_hyperparams/%s_hyperparams.pickle' % model_name, 'wb') as handle:
            pickle.dump(winning_hyperparameters, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def cross_validation_grid_ngboost(self, possible_hyperparams):
        def get_param_grid(dicts):
            return [dict(zip(dicts.keys(), p)) for p in it.product(*dicts.values())]

        model_name = 'ng_boost'

        # training and testing data if we have single target variable
        X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        if not self.nosmote:
            print('\nSize of the data before Oversampling: %d\n' % len(y_train))

        self.len_train_before = len(y_train)

        # indices of the rare values
        rare_values = self.rtrain
        y_train_old = y_train
        rare_vec = [1 if i in rare_values else 0 for i in range(len(y_train))]

        y_train = np.array(rare_vec)

        tempModels = []

        # specify the type of cv (kfold vs. repeated kfold)
        if self.nb_repeats_grid is None:
            print('running Stratified %d-fold cross validation' % self.nb_folds_grid)
            kf = StratifiedKFold(n_splits=self.nb_folds_grid)
        else:
            print('running Repeated Stratified %d-fold-%d-repeats cross validation' % (
                self.nb_folds_grid, self.nb_repeats_grid))
            kf = RepeatedStratifiedKFold(n_splits=self.nb_folds_grid, n_repeats=self.nb_repeats_grid,
                                         random_state=2652124)

        t1 = time.time()

        parameters = possible_hyperparams

        # hyper parameters loop
        t1 = time.time()
        print('Total nb of hyper params: %d' % len(get_param_grid(parameters)))
        for parameter in get_param_grid(parameters):

            model = NGBRegressor(**parameter)

            ngb_nll = []
            r2_scores, adj_r2_scores, rmse_scores, mse_scores, mae_scores, mape_scores = [], [], [], [], [], []
            f1_scores, f2_scores, f05_scores, prec_scores, rec_scores = [], [], [], [], []

            count = 0
            for train_index, test_index in kf.split(X_train, y_train):
                X_train_inner, X_val = X_train[train_index], X_train[test_index]

                # replace the y's with their original values after splitting in a stratified manner
                y_train_inner, y_val = y_train_old[train_index], y_train_old[test_index]

                if self.scale:
                    X_train_inner, X_val, y_train_inner, y_val, scaler_out_final = self.scale_cols(X_train_inner, X_val,
                                                                                                   y_train_inner, y_val)

                model.fit(X_train_inner, y_train_inner)

                # point and probabilistic predictions
                y_pred = model.predict(X_val)
                y_forecast = model.pred_dist(X_val)

                # AFTER PREDICTION, reverse the scaled output (if self.scale_output is on).
                # The idea is to reverse scaling JUST BEFORE printing out the error metrics
                if self.scale_output:
                    if self.output_zscore or self.output_minmax:
                        y_pred = scaler_out_final.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
                        y_val = scaler_out_final.inverse_transform(y_val.reshape(-1, 1)).reshape(-1)
                    elif self.output_log:
                        y_pred = np.exp(y_pred)
                        y_val = np.exp(y_val)
                    else:
                        y_pred = self.inverse_boxcox(y_pred, self.y_train_lambda_)
                        y_val = self.inverse_boxcox(y_val, self.y_test_lambda_)

                r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance, rmetrics = get_stats(y_val, y_pred,
                                                                                                    X_val.shape[1],
                                                                                                    self.thr_rel,
                                                                                                    self.phi_params,
                                                                                                    self.loss_params)

                # add point predictions score
                r2_scores.append(r2)
                adj_r2_scores.append(adj_r2)
                rmse_scores.append(rmse)
                mse_scores.append(mse)
                mae_scores.append(mae)
                mape_scores.append(mape)
                f1_scores.append(rmetrics['ubaF1'][0])
                f2_scores.append(rmetrics['ubaF2'][0])
                f05_scores.append(rmetrics['ubaF05'][0])
                prec_scores.append(rmetrics['ubaprec'][0])
                rec_scores.append(rmetrics['ubarec'][0])

                # add the ngb probabilistic predictions
                ngb_nll.append(-y_forecast.logpdf(y_val.flatten()).mean())

                count += 1

            tempModels.append(
                [parameter, np.mean(r2_scores), np.mean(adj_r2_scores), np.mean(rmse_scores), np.mean(mse_scores),
                 np.mean(mae_scores), np.mean(mape_scores),
                 np.mean(f1_scores), np.mean(f2_scores), np.mean(f05_scores),
                 np.mean(prec_scores), np.mean(rec_scores),
                 np.mean(ngb_nll), np.std(ngb_nll)])

        tempModels = sorted(tempModels, key=lambda k: k[3])
        winning_hyperparameters = tempModels[0][0]
        
        self.nlls['ng_boost'] = {
            'mean': tempModels[0][12],
            'std': tempModels[0][13]
        }
        print('winning hyper parameters: ', str(winning_hyperparameters))
        print('Best Validation Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
              'F1: %.5f\nF2: %.5f\nF0.5: %.5f\nprec: %.5f\nrec: %.5f\nNGB-NLL: %.5f +/- %.5f\n' %
              (tempModels[0][1], tempModels[0][2], tempModels[0][3], tempModels[0][4], tempModels[0][5],
               tempModels[0][6],
               tempModels[0][7], tempModels[0][8], tempModels[0][9], tempModels[0][10], tempModels[0][11],
               tempModels[0][12], tempModels[0][13]))
        
        if self.save_errors_xlsx:
            if self.save_validation:
                self.results.loc['%s_val' % model_name] = pd.Series({
                    'r2': tempModels[0][1],
                    'adj-r2': tempModels[0][2],
                    'rmse': tempModels[0][3],
                    'mse': tempModels[0][4],
                    'mae': tempModels[0][5],
                    'mape': tempModels[0][6],
                    'F1': tempModels[0][7],
                    'F2': tempModels[0][8],
                    'F05': tempModels[0][9],
                    'prec': tempModels[0][10],
                    'rec': tempModels[0][11],
                    'nll': tempModels[0][12]
                })

        model = NGBRegressor(**winning_hyperparameters)

        # replace back y_train to the old one after finishing stratified k fold cross validation
        y_train = y_train_old

        if self.scale:
            X_train, X_test, y_train, y_test, scaler_out_final = self.scale_cols(X_train, X_test, y_train, y_test)

        if not self.nosmote:
            # apply over/under sampling only on training data

            df_combined = self.prepare_data(X_train, y_train)

            # transform one hot encodings from float to int
            if self.one_hot_encoded is not None:
                df_combined = convert_cols_int(df_combined, self.one_hot_encoded)

            if self.smogn:
                oversampled, X_train, y_train = self.apply_smogn(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            elif self.smoter:
                oversampled, X_train, y_train = self.apply_smoter(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            elif self.gn:
                oversampled, X_train, y_train = self.apply_gn(df_combined, plotdensity=True, model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            else:
                oversampled, X_train, y_train = self.apply_rand_under(df_combined, plotdensity=True,
                                                                      model_name=model_name)
                if self.one_hot_encoded is not None:
                    self.count_abnormal(oversampled)
                    oversampled = self.round_oversampled_one_hot_encoded(oversampled)
                self.save_oversampled_dataset(oversampled, model_name)

            # show plot of target variable after oversampling
            plot_target_variable(oversampled, self.target_variable, self.output_folder + 'plots/',
                                 '%s_oversampled_%s_%s' % (
                                     self.target_variable, get_method(self.smogn, self.smoter, self.rand_under,
                                                                      self.gn), model_name))

            print('\n ********************* Rare Values - After SMOTE ********************* ')
            yrelafter, distances = get_relevance_oversampling(oversampled, self.target_variable, self.targetrel)
            roversampled = get_rare_indices(oversampled[self.target_variable], yrelafter, self.thr_rel,
                                            self.phi_params['control.pts'])

            # save rare indices of the oversampled <<training>> data - i.e. After Oversampling
            self.save_rare_after_oversampling(roversampled, model_name)

            avg_dist = np.mean(distances)

            self.rare_statistics.loc[model_name] = pd.Series({
                'total_rare_before': (len(self.rtrain + self.rtest) / (len(self.df_train) + len(self.df_test))) * 100,
                'rare_train_before': (len(self.rtrain) / len(self.df_train)) * 100,
                'rare_test_before': (len(self.rtest) / len(self.df_test)) * 100,
                'train_size_before': self.len_train_before,
                'train_size_after': len(oversampled),
                'rare_train_after': (len(roversampled) / len(oversampled)) * 100,
                'Cperc': self.Cperc,
                'avg_dist': avg_dist
            })

            # plots the rare indices
            plot_rare(oversampled[self.target_variable], roversampled, self.target_variable,
                      output_folder=self.output_folder + 'plots/',
                      fig_name='rare_values_after_%s' % get_method(self.smogn, self.smoter, self.rand_under, self.gn),
                      model_name=model_name)

            print('Average distances between actual and synthesized: {}'.format(avg_dist))

        model.fit(X_train, y_train)

        # end training time
        t2 = time.time()
        time_taken_min = float(t2 - t1) / 60
        time_taken_sec = float(t2 - t1)

        # save the model
        models_folder = self.output_folder + 'trained_models/'
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        pkl_filename = "%s.pkl" % model_name
        with open(models_folder + pkl_filename, 'wb') as file:
            pickle.dump(model, file)
        print('saved model to {} as {}.pkl'.format(models_folder, model_name))

        if not self.nosmote:
            print('\nSize of the data after Oversampling: %d\n' % len(y_train))

        # the point and probabilistic predictions
        y_pred = model.predict(X_test)
        forecast = model.pred_dist(X_test)

        # the NGB NLL & RMSE
        ngb_nll_test = -forecast.logpdf(y_test.flatten()).mean()

        # testing_nonrare = list(set(list(range(0, len(self.X_test)))) - set(self.rtest))
        # if len(self.rtest) > 5:
        #     candidates_rare = np.random.choice(self.rtest, 5)
        #     candidates_nonrare = np.random.choice(testing_nonrare, 5)
        # else:
        #     # choose only 3 if there are very few rare in testing (which is not the case usually)
        #     candidates_rare = np.random.choice(self.rtest, 3)
        #     candidates_nonrare = np.random.choice(testing_nonrare, 3)
        #
        # self.plot_specific_probas(model, X_test, y_test,
        #                     candidates_rare, candidates_nonrare,
        #                     cands1_lab='rare', cands2_lab='non rare',
        #                     output_folder=os.path.join(self.output_folder, 'probabilistic_forecasts'),
        #                     fig_name='rare_nonrare_ngb')

        candidates_rare = self.rtest
        candidates_nonrare = list(set(list(range(0, len(self.X_test)))) - set(self.rtest))

        self.plot_specific_probas(model, X_test, y_test,
                            candidates_rare, candidates_nonrare,
                            cands1_lab='rare', cands2_lab='non rare',
                            output_folder=os.path.join(self.output_folder, 'probabilistic_forecasts'))

        if self.scale_output:
            if self.output_log:
                y_pred_reverse = np.exp(y_pred.reshape(-1, 1))
                y_test = np.exp(y_test)
            elif self.output_box:
                y_pred_reverse = self.inverse_boxcox(y_pred, self.y_train_lambda_)
                y_test = self.inverse_boxcox(y_test, self.y_test_lambda_)
            else:
                # minmax or z-score
                y_pred_reverse = scaler_out_final.inverse_transform(y_pred.reshape(-1, 1)).reshape(-1)
                y_test = scaler_out_final.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

            # the output dataset must contain the
            output_dataset = self.create_output_dataset(y_pred_reverse, model_name,
                                                        self.output_folder + 'output_vector_datasets/')
        else:
            output_dataset = self.create_output_dataset(y_pred, model_name,
                                                        self.output_folder + 'output_vector_datasets/')

        self.plot_actual_vs_predicted(y_test, y_pred,
                                      self.output_folder + 'train_test_forecasts_lineplot/',
                                      model_name)

        self.plot_actual_vs_predicted_scatter_bisector(y_test, y_pred,
                                                       self.output_folder + 'train_test_forecasts_scatterplot_bisector/',
                                                       model_name)

        r2, adj_r2, rmse, mse, mae, mape, pearson, spearman, distance, rmetrics = get_stats(y_test, y_pred,
                                                                                            X_test.shape[1],
                                                                                            self.thr_rel,
                                                                                            self.phi_params,
                                                                                            self.loss_params)
        print('\nTesting Scores:\nR^2: %.5f\nAdj R^2: %.5f\nRMSE: %.5f\nMSE: %.5f\nMAE: %.5f\nMAPE: %.5f\n'
              'F1: %.5f\nF2: %.5f\nF0.5: %.5f\nprec: %.5f\nrec: %.5f\nNGB-NLL: %.5f\n' %
              (r2, adj_r2, rmse, mse, mae, mape,
               rmetrics['ubaF1'][0], rmetrics['ubaF2'][0], rmetrics['ubaF05'][0], rmetrics['ubaprec'][0],
               rmetrics['ubarec'][0], ngb_nll_test))

        avg_target = np.mean(y_test)
        print('Average %s: %.5f' % (self.target_variable, avg_target))
        print('Pearson Correlation: %.5f' % pearson)
        print('Spearman Correlation: %.5f' % spearman)
        print('Distance Correlation: %.5f\n' % distance)

        print('function took %.5f mins\nfunction took %.5f secs\n' % (time_taken_min, time_taken_sec))

        if self.save_errors_xlsx:
            if self.save_validation:
                row_name = '%s_test' % model_name
            else:
                row_name = model_name

            self.results.loc[row_name] = pd.Series({'r2': r2, 'adj-r2': adj_r2, 'rmse': rmse, 'mse': mse,
                                                    'mae': mae, 'mape': mape,
                                                    'F1': rmetrics['ubaF1'][0],
                                                    'F2': rmetrics['ubaF2'][0],
                                                    'F05': rmetrics['ubaF05'][0],
                                                    'prec': rmetrics['ubaprec'][0],
                                                    'rec': rmetrics['ubarec'][0],
                                                    'nll': ngb_nll_test,
                                                    'winning_hyperparams': str(winning_hyperparameters),
                                                    'avg_%s' % self.target_variable: avg_target,
                                                    'pearson': pearson, 'spearman': spearman,
                                                    'distance': distance,
                                                    'train_time_min': time_taken_min,
                                                    'train_time_sec': time_taken_sec})
        
        self.nlls['ng_boost']['test'] = ngb_nll_test
        
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
            df_test_curr.to_csv(output_folder + '%s_%s/test_%s.csv' % (self.service_name, self.mohafaza, model_name),
                                index=False)

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

                X_train_boxscaled = np.array(
                    [list(scipy.stats.boxcox(X_train[:, self.idx_box[i]])[0]) for i in range(len(self.idx_box))]).T
                X_test_boxscaled = np.array(
                    [list(scipy.stats.boxcox(X_test[:, self.idx_box[i]])[0]) for i in range(len(self.idx_box))]).T

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
                y_train, self.y_train_lambda_ = scipy.stats.boxcox(y_train)
                y_test, self.y_test_lambda_ = scipy.stats.boxcox(y_test)

            else:
                if self.output_log:
                    y_train = np.log(y_train)
                    y_test = np.log(y_test)

        return X_train, X_test, y_train, y_test, scaler_out_final

    def produce_learning_curve(self, X_train, y_train, model, model_name, nb_splits, output_folder, parameters,
                               nb_repeats=None):

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

        # X_train, y_train = self.X_train, self.y_train
        pipe = None
        if self.scale:
            if self.scale_output:
                if self.output_zscore:
                    scaler = StandardScaler()
                    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
                elif self.output_minmax:
                    scaler = MinMaxScaler()
                    y_train = scaler.fit_transform(y_train.reshape(-1, 1)).reshape(-1)
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

                    X_train_boxscaled = np.array(
                        [list(scipy.stats.boxcox(X_train[:, self.idx_box[i]])[0]) for i in range(len(self.idx_box))]).T

                    for i in range(len(self.idx_box)):
                        X_train[:, self.idx_box[i]] = X_train_boxscaled[:, i]

                if self.input_log is not None:
                    # apply Log transform to the specified columns.
                    if X_train.dtype != 'float64':
                        X_train = X_train.astype('float64')

                    X_train_logscaled = np.log(X_train[:, self.idx_log])

                    for i in range(len(self.idx_log)):
                        X_train[:, self.idx_log[i]] = X_train_logscaled[:, i]

        train_sizes, train_scores, test_scores = learning_curve(pipe, X_train, y_train, cv=cv,
                                                                scoring='neg_mean_squared_error')  # calculate learning curve values

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

    def mkdir(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

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

    def save_uncertainties(self):
        dest = os.path.join(self.output_folder, 'uncertainty_scores')
        self.mkdir(dest)
        with open(os.path.join(dest, 'uncertainty_scores.p'), 'wb') as fp:
            pickle.dump(self.uncertainties, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
        ''' saves the error metrics (stored in `results`) as a csv file and the NLL scores'''

        # saving the error metrics
        if self.results is not None:
            errors_df = self.results
            path = self.output_folder + 'error_metrics_csv/'
            if not os.path.exists(path):
                os.makedirs(path)
            errors_df.to_csv(path + 'errors.csv')

        # saving the NLL scores
        self.save_nlls()

        # plot the NLL scores
        self.show_nll_scores()

        # saving the uncertainty scores
        self.save_uncertainties()

        # plot the uncertainty scores
        self.show_uncertainty_scores()

    def rare_statistics_to_csv(self):
        ''' saves statistics about rare values in the data (stored in `rare_statistics`) as a csv file '''
        statistics_df = self.rare_statistics
        path = self.output_folder + 'rare_statistics_csv/'
        if not os.path.exists(path):
            os.makedirs(path)
        statistics_df.to_csv(path + 'statistics.csv')


def mean_absolute_percentage_error(y_true, y_pred):
    '''
    Function to compute the mean absolute percentage error (MAPE) between an actual and
    predicted vectors
    :param y_true: the actual values
    :param y_pred: the predicted values
    :return: MAPE
    '''
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def get_stats(y_test, y_pred, nb_columns, thr_rel, phi_params, loss_params):
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

    r2_Score = r2_score(y_test, y_pred)  # r-squared
    adjusted_r2 = 1 - ((1 - r2_Score) * (n - 1)) / (n - nb_columns - 1)  # adjusted r-squared
    rmse_score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
    mse_score = mean_squared_error(y_test, y_pred)  # MSE
    mae_score = mean_absolute_error(y_test, y_pred)  # MAE
    mape_score = mean_absolute_percentage_error(y_test, y_pred)  # MAPE

    trues = np.array(y_test)
    preds = np.array(y_pred)

    method = phi_params['method']
    npts = phi_params['npts']
    controlpts = phi_params['control.pts']
    ymin = loss_params['ymin']
    ymax = loss_params['ymax']
    tloss = loss_params['tloss']
    epsilon = loss_params['epsilon']

    rmetrics = runit.eval_stats(trues, preds, thr_rel, method, npts, controlpts, ymin, ymax, tloss, epsilon)

    # create a dictionary of the r metrics extracted above
    rmetrics_dict = dict(zip(rmetrics.names, list(rmetrics)))

    if isinstance(y_pred[0], np.ndarray):
        y_pred_new = [x[0] for x in y_pred]
        y_pred = y_pred_new
    pearson_corr, _ = pearsonr(y_test, y_pred)
    spearman_corr, _ = spearmanr(y_test, y_pred)
    distance_corr = distance.correlation(y_test, y_pred)

    return r2_Score, adjusted_r2, rmse_score, mse_score, mae_score, mape_score, pearson_corr, spearman_corr, distance_corr, rmetrics_dict
