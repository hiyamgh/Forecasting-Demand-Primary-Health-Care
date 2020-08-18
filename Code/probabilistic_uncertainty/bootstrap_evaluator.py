import numpy as np
from Code.probabilistic_uncertainty.bootstrap_trainer import BootstrapTrainer
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')


def bootstrap_evaluation(X_train, y_train, X_test, y_test, dropout, learning_rate, epochs, n_heads):

    # initialize trainable model
    model = BootstrapTrainer(X_train=X_train, y_train=y_train, dropout=dropout, learning_rate=learning_rate,
                             epochs=epochs, n_heads=n_heads, display_step=10)

    # training
    sess, x_placeholder, dropout_placeholder, mask_placeholder = model.train_model()

    # evaluation
    feed_dict = {model.input_data: X_test,
                 model.target_data: y_test.reshape(-1, 1) if y_test.ndim == 1 else y_test,
                 model.dropout_placeholder: 0,
                 model.mask_placeholder: np.ones(shape=(len(X_test), n_heads, 1))}

    # evaluation
    y_eval, uncertainties_eval, heads_eval, loss = sess.run([model.mean, model.variance, model.heads, model.loss], feed_dict)

    heads_eval = np.array(heads_eval).reshape(len(X_test), n_heads)

    return y_eval, uncertainties_eval, loss


if __name__ == "__main__":
    KEY = 'all_columns'
    # path = datasets[KEY]
    path = '../../input/all_without_date/collated/all_columns/'
    df_train = pd.read_csv(path + 'df_train_collated.csv')
    df_test = pd.read_csv(path + 'df_test_collated.csv')
    target_variable = 'demand'

    # numpy arrays X_train, y_train, X_test, y_test
    X_train = np.array(df_train.loc[:, df_train.columns != target_variable])
    y_train = np.array(df_train.loc[:, target_variable])

    X_test = np.array(df_test.loc[:, df_test.columns != target_variable])
    y_test = np.array(df_test.loc[:, target_variable])

    heads = [5]
    for n_heads in heads:
        print('====================================== HEADS: {} ======================================'.format(n_heads))
        y_eval, uncertainty, loss = bootstrap_evaluation(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dropout=0.3,
                                                 learning_rate=1e-3, epochs=100, n_heads=n_heads)

        # print('uncertainty: {}'.format(uncertainty))
        print('loss: {}'.format(loss))

