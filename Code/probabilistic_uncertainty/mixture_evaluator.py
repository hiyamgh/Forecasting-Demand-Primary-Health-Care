import numpy as np
import pandas as pd
from Code.probabilistic_uncertainty.mixture_trainer import MixtureTrainer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')


def mixture_evaluation(X_train, y_train, X_test, y_test, dropout, learning_rate, epochs, n_mixtures):

    # initialize trainable model
    model = MixtureTrainer(X_train=X_train, y_train=y_train, dropout=dropout, learning_rate=learning_rate,
                           epochs=epochs, n_mixtures=n_mixtures, display_step=10)

    # training
    sess, x_placeholder, dropout_placeholder = model.train_model()

    # evaluation
    feed_dict = {model.input_data: X_test,
                 model.target_data: y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test,
                 dropout_placeholder: 0}

    y_eval, uncertainties_eval, loss = sess.run([model.mean, model.uncertainties, model.loss], feed_dict)

    aleatoric_eval, epistemic_eval = uncertainties_eval[0], uncertainties_eval[1]
    total_uncertainty_eval = aleatoric_eval + epistemic_eval

    return y_eval, aleatoric_eval, epistemic_eval, total_uncertainty_eval, loss


if __name__ == "__main__":
    path = '../../input/all_without_date/collated/all_columns/'
    df_train = pd.read_csv(path + 'df_train_collated.csv')
    df_test = pd.read_csv(path + 'df_test_collated.csv')
    target_variable = 'demand'

    # numpy arrays X_train, y_train, X_test, y_test
    X_train = np.array(df_train.loc[:, df_train.columns != target_variable])
    y_train = np.array(df_train.loc[:, target_variable])

    X_test = np.array(df_test.loc[:, df_test.columns != target_variable])
    y_test = np.array(df_test.loc[:, target_variable])

    mixture_values = [1]
    for n_mixtures in mixture_values:
        print(
            '====================================== N_MIXTURES: {} ======================================'.format(n_mixtures))
        y_eval, epistemic_eval, aleatoric_eval, total_uncertainty_eval, loss = mixture_evaluation(X_train=X_train, y_train=y_train, X_test=X_test,
                                                                                                  y_test=y_test, dropout=0.2,
                                                 learning_rate=1e-3, epochs=100, n_mixtures=n_mixtures)
        # print('uncertainty: {}'.format(uncertainty))
        print('loss: {}'.format(loss))





