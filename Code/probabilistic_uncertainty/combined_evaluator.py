import numpy as np
import pandas as pd
from Code.probabilistic_uncertainty.combined_trainer import CombinedTrainer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')


def combined_evaluation(X_train, y_train, X_test, y_test, dropout, learning_rate, epochs, n_passes):

    # initialize trainable model
    model = CombinedTrainer(X_train=X_train, y_train=y_train, dropout=dropout, learning_rate=learning_rate, epochs=epochs, display_step=10)

    # training
    sess, x_placeholder, dropout_placeholder = model.train_model()

    # evaluation
    feed_dict = {model.input_data: X_test,
                 model.target_data: y_test.reshape([-1, 1]) if y_test.ndim == 1 else y_test,
                 model.dropout_placeholder: dropout}

    predictions = []
    aleatorics = []
    losses = []

    for _ in range(n_passes):
        prediction, aleatoric, loss = sess.run([model.prediction, model.log_variance, model.loss], feed_dict)
        predictions.append(prediction.flatten())
        aleatorics.append(aleatoric.flatten())
        losses.append(loss)

    # predictions and uncertainties
    y_eval = np.mean(predictions, axis=0).flatten()
    epistemic_eval = np.var(predictions, axis=0).flatten()
    aleatoric_eval = np.mean(aleatorics, axis=0).flatten()
    total_uncertainty_eval = epistemic_eval + aleatoric_eval
    loss_eval = np.mean(losses)

    return y_eval, epistemic_eval, aleatoric_eval, total_uncertainty_eval, loss_eval


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

    dropout_values = [0.3]
    for dropout in dropout_values:
        print('====================================== DROPOUT: {} ======================================'.format(dropout))
        y_eval, epistemic_eval, aleatoric_eval, total_uncertainty_eval, loss = combined_evaluation(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, dropout=0.3,
                                                 learning_rate=1e-3, epochs=100, n_passes=100)

        print('Loss: {}'.format(loss))
        print('y_eval.shape: {}'.format(y_eval.shape))
