import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from DeepEnsemble import DeepEnsembleModel
import random
import os


def making_batch(data_size, sample_size, data_x, data_y):
    # data_size: number of rows
    # sample_size: desired batch size
    # num_data: number of input columns

    # Making batches(testing)
    batch_idx = np.random.choice(data_size, sample_size)

    batch_x = np.zeros([sample_size, data_x.shape[1]])
    batch_y = np.zeros([sample_size, 1])

    for i in range(batch_idx.shape[0]):
        # print(batch_idx[i])
        batch_x[i, :] = data_x[batch_idx[i], :]
        batch_y[i] = data_y[batch_idx[i]]

    return batch_x, batch_y


def ensemble_mean_var(ensemble, xs, ys, sess):
    en_mean = 0
    en_var = 0
    en_nll = 0

    outputs_per_ensemble = []
    for model in ensemble:
        feed = {model.input_data: xs, model.target_data: ys}
        mean, var, nll = sess.run([model.output_mu, model.output_sig_pos, model.loss], feed)
        outputs_per_ensemble.append(mean)
        en_mean += mean
        en_var += var + mean**2
        en_nll += nll

    en_mean /= len(ensemble)
    en_var /= len(ensemble)
    en_var -= en_mean**2
    en_nll /= len(ensemble)

    return en_mean, en_var, en_nll, outputs_per_ensemble


# ensemble, sizes, max_iter, X, y, batch_size
def train_ensemble(X_train, y_train, X_test, y_test, learning_rate, max_iter, batch_size, sizes, optimizer_name, produce_plots=False, output_folder=None, fig_name=None):

    # get the ensemble of networks
    ensemble = [DeepEnsembleModel(sizes, num_data=X_train.shape[1], model_scope='model' + str(i), learning_rate=learning_rate,
                                  max_iter=max_iter, batch_size=batch_size, optimizer_name=optimizer_name) for i in range(len(sizes))]
    # Create Session
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        # I think this iteration resembles the number of epochs
        for iter in range(max_iter):

            # get the output from each network
            for model in ensemble:

                # in each network, we train and test, get the output
                batch_x, batch_y = making_batch(X_train.shape[0], batch_size, X_train, y_train)

                feed = {model.input_data: batch_x, model.target_data: batch_y}
                _, nll, m, v = sess.run([model.train_opt, model.loss, model.output_mu, model.output_sig_pos], feed)
                print('output_sig_pos: {}\noutput_mu: {}'.format(np.any(np.isnan(v)), np.any(np.isnan(m))))

                # print('training loss: {}'.format(nll))

                if np.any(np.isnan(nll)):
                    print('There is Nan in loss')

            if iter % 10 == 0 and iter != 0:
                print('itr: {}, nll: {}'.format(iter, nll))

        mean, var, nll, outputs_per_ensemble = test_ensemble(ensemble, sess, X_test, y_test, produce_plots, output_folder, fig_name)
        return mean, var, nll, outputs_per_ensemble


def check_create_output_folder(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


def plot_ensemble_actual_vs_predicted(y_true, y_pred, upper, lower, output_folder, fig_name, outputspe, randrangestart=None):
    if randrangestart is None:
        if outputspe is not None:
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, len(outputspe))]
            for i, out in enumerate(outputspe):
                plt.plot(list(range(len(y_true))), out, label='out{}'.format(i+1), color= colors[i])
        plt.plot(list(range(len(y_true))), y_true, 'b-', label='actual/mean')
        plt.plot(list(range(len(y_true))), y_pred, 'r-', label='predicted')
        plt.fill_between(list(range(len(y_true))), lower[:, 0], upper[:, 0], color='yellow', alpha=0.2)
        plt.legend()

        check_create_output_folder(output_folder)
        plt.savefig(os.path.join(output_folder, fig_name + '_all'))
        plt.close()
    else:
        start = randrangestart
        end = start + 10
        if outputspe is not None:
            cmap = plt.get_cmap('gnuplot')
            colors = [cmap(i) for i in np.linspace(0, 1, len(outputspe))]
            for i, out in enumerate(outputspe):
                print('processing out{}'.format(i))
                plt.plot(list(range(start, end)), out[start: end, :], label='out{}'.format(i+1), color= colors[i])
        plt.plot(list(range(start, end)), y_true[start: end], 'b-', label='actual')
        plt.plot(list(range(start, end)), y_pred[start: end, :], 'r-', label='predicted')
        plt.fill_between(list(range(start, end)), lower[start: end, 0], upper[start: end, 0], color='yellow', alpha=0.2)
        plt.legend()
        check_create_output_folder(output_folder)
        plt.savefig(os.path.join(output_folder, fig_name + '_rand'))
        plt.close()


def test_ensemble(ensemble, sess, X_test, y_test, produce_plots, output_folder, fig_name):
    y_testtemp = y_test.reshape(-1, 1)
    mean, var, nll, outputs_per_ensemble = ensemble_mean_var(ensemble, X_test, y_testtemp, sess)
    std = np.sqrt(var)
    upper = mean + 3 * std
    lower = mean - 3 * std

    # I will produce two plots, one on the whole testing
    # another on a random range of 10 continuous testing points
    if produce_plots:
        rnum = random.randint(0, len(y_test) - 10)

        plot_ensemble_actual_vs_predicted(y_true=y_test, y_pred=mean, upper=upper, lower=lower, output_folder=output_folder, fig_name=fig_name,
                                          outputspe=outputs_per_ensemble, randrangestart=None)
        plot_ensemble_actual_vs_predicted(y_true=y_test, y_pred=mean, upper=upper, lower=lower, output_folder=output_folder, fig_name=fig_name,
                                          outputspe=outputs_per_ensemble, randrangestart=rnum)

    return mean, var, nll, outputs_per_ensemble




