# Copyright 2016, Yarin Gal, All rights reserved.
# This code is based on the code by Jose Miguel Hernandez-Lobato used for his
# paper "Probabilistic Backpropagation for Scalable Learning of Bayesian Neural Networks".

import warnings

warnings.filterwarnings("ignore")

import math
from scipy.special import logsumexp
import numpy as np

from keras.regularizers import l2
from keras import Input
from keras.layers import Dropout
from keras.layers import Dense
from keras import Model

import time


class net:

    def __init__(self, X_train, y_train, n_hidden, n_epochs=40,
                 normalize=False, tau=1.0, dropout=0.05, T=1000):

        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param X_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Numer of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unles the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
            @param tau          Tau value used for regularization
            @param dropout      Dropout rate for all the dropout layers in the
                                network.
            @param T            Number of stochastic passes over the network
        """

        # We normalize the training data to have zero mean and unit standard
        # deviation in the training set if necessary

        if normalize:
            self.std_X_train = np.std(X_train, 0)
            self.std_X_train[self.std_X_train == 0] = 1
            self.mean_X_train = np.mean(X_train, 0)
        else:
            self.std_X_train = np.ones(X_train.shape[1])
            self.mean_X_train = np.zeros(X_train.shape[1])

        X_train = (X_train - np.full(X_train.shape, self.mean_X_train)) / \
                  np.full(X_train.shape, self.std_X_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        # Normalize training data

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train
        y_train_normalized = np.array(y_train_normalized, ndmin=2).T

        # We construct the network
        N = X_train.shape[0]
        batch_size = 128
        lengthscale = 1e-2
        reg = lengthscale ** 2 * (1 - dropout) / (2. * N * tau)

        inputs = Input(shape=(X_train.shape[1],))
        inter = Dropout(dropout)(inputs, training=True)
        inter = Dense(n_hidden[0], activation='relu', W_regularizer=l2(reg))(inter)
        for i in range(len(n_hidden) - 1):
            inter = Dropout(dropout)(inter, training=True)
            inter = Dense(n_hidden[i + 1], activation='relu', W_regularizer=l2(reg))(inter)
        inter = Dropout(dropout)(inter, training=True)
        outputs = Dense(y_train_normalized.shape[1], W_regularizer=l2(reg))(inter)
        model = Model(inputs, outputs)

        model.compile(loss='mean_squared_error', optimizer='adam')

        # We iterate the learning process
        start_time = time.time()
        model.fit(X_train, y_train_normalized, batch_size=batch_size, nb_epoch=n_epochs, verbose=0)
        self.model = model
        self.tau = tau
        self.T = T
        self.running_time = time.time() - start_time

        # We are done!

    def predict(self, X_test, y_test):

        """
            Function for making predictions with the Bayesian neural network.

            @param X_test   The matrix of features for the test data


            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.

        """

        X_test = np.array(X_test, ndmin=2)
        y_test = np.array(y_test, ndmin=2).T

        # We normalize the test set

        X_test = (X_test - np.full(X_test.shape, self.mean_X_train)) / \
                 np.full(X_test.shape, self.std_X_train)

        # We compute the predictive mean and variance for the target variables
        # of the test data

        model = self.model

        # standard_pred are the point predictions
        standard_pred = model.predict(X_test, batch_size=500, verbose=1)
        standard_pred = standard_pred * self.std_y_train + self.mean_y_train

        # rmse_standard_pred is the RMSE of the point predictions
        rmse_standard_pred = np.mean((y_test.squeeze() - standard_pred.squeeze()) ** 2.) ** 0.5

        # Let's make the MC dropout predictions. Here we get the uncertainty.
        # Let's choose how many times we predict a single point (how many stochastic passes over the network)
        # T = 1000

        # Here in Yt_hat we store the results for each pass
        Yt_hat = np.array([model.predict(X_test, batch_size=500, verbose=0) for _ in range(self.T)])

        # Un-normalize
        Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train

        # Calculate the predictive mean
        MC_pred = np.mean(Yt_hat, 0)

        # Calculate the predictive variance
        predictive_variance = np.var(Yt_hat, axis=0)
        # Hiyam: the predictive variance is actually the uncertainty estimate in the MC dropout

        # this is like NGB-RMSE ==> RMSE of the probabilistic distributions
        rmse = np.mean((y_test.squeeze() - MC_pred.squeeze()) ** 2.) ** 0.5

        # We compute the test log-likelihood
        ll = (logsumexp(-0.5 * self.tau * (y_test[None] - Yt_hat) ** 2., 0) - np.log(self.T)
              - 0.5 * np.log(2 * np.pi) + 0.5 * np.log(self.tau))
        test_ll = np.mean(ll)

        y_pred = standard_pred.squeeze()

        # I have put this here, the negative log likelihood is the negative of the log likelihood
        test_nll = - test_ll
        MC_nll = test_nll

        # return y_pred, Yt_hat, MC_nll, predictive_variance (uncertainty)
        return y_pred, Yt_hat, MC_nll, predictive_variance
