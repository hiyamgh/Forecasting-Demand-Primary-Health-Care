from Code.probabilistic_uncertainty import mixture_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class MixtureTrainer:

    def __init__(self, X_train, y_train, dropout, learning_rate, epochs, n_mixtures, display_step=10):

        tf.reset_default_graph()

        # training input & output
        self.X_train = X_train
        self.y_train = y_train

        # other variables
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_mixtures = n_mixtures
        self.display_step = display_step

        # for multivariate data
        self.input_data = tf.placeholder(tf.float64, [None, self.X_train.shape[1]])
        self.target_data = tf.placeholder(tf.float64, [None, 1])

        self.dropout_placeholder = tf.placeholder(tf.float64)
        self.eps = 1e-4

        self.gmm, self.mean, self.uncertainties = mixture_model.mixture_model(self.input_data, self.dropout_placeholder,
                                                               n_mixtures=self.n_mixtures)

        mixture_weights = self.gmm[0]
        mixture_means = self.gmm[1]
        mixture_variances = self.gmm[2]

        dist = tf.distributions.Normal(loc=mixture_means, scale=mixture_variances)

        self.loss = - tf.reduce_mean(
            tf.log(tf.reduce_sum(mixture_weights * dist.prob(self.target_data), axis=1) + self.eps),
            axis=0
        )

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(init)

    def train_model(self):
        for epoch in range(self.epochs):
            feed_dict = {self.input_data: self.X_train,
                         self.target_data: self.y_train.reshape([-1, 1] if self.y_train.ndim == 1 else self.y_train),
                         self.dropout_placeholder: self.dropout}

            self.sess.run(self.train, feed_dict=feed_dict)

            if epoch % self.display_step == 0:
                print("Epoch {}".format(epoch))
                current_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                print("Loss {}".format(current_loss))
                print("================")

        print("Training done")

        return self.sess, self.input_data, self.dropout_placeholder


# def mixture_training(x_truth, y_truth, dropout, learning_rate, epochs, n_mixtures, display_step=10):
#     """
#     Generic training of a Mixture Density Mixture Network for 2D data.
#
#     :param x_truth: training samples x
#     :param y_truth: training samples y / label
#     :param dropout:
#     :param learning_rate:
#     :param epochs:
#     :param n_mixtures: Number of mixtures in GMM
#     :param display_step:
#     :return: session, x_placeholder, dropout_placeholder
#     """
#     tf.reset_default_graph()
#
#     # for multivariate data
#     x_placeholder = tf.placeholder(tf.float64, [None, x_truth.shape[1]])
#     y_placeholder = tf.placeholder(tf.float64, [None, 1])
#
#     dropout_placeholder = tf.placeholder(tf.float64)
#     eps = 1e-4
#
#     gmm, mean, uncertainties = mixture_model.mixture_model(x_placeholder, dropout_placeholder, n_mixtures=n_mixtures)
#
#     tf.add_to_collection("gmm", gmm)
#     tf.add_to_collection("prediction", mean)
#     tf.add_to_collection("uncertainties", uncertainties)
#
#     mixture_weights = gmm[0]
#     mixture_means = gmm[1]
#     mixture_variances = gmm[2]
#
#     dist = tf.distributions.Normal(loc=mixture_means, scale=mixture_variances)
#     loss = - tf.reduce_mean(
#         tf.log(tf.reduce_sum(mixture_weights * dist.prob(y_placeholder), axis=1) + eps),
#         axis=0
#     )
#
#     # tf.add_to_collection("loss", loss)
#
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     train = optimizer.minimize(loss)
#
#     init = tf.global_variables_initializer()
#     sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#     sess.run(init)
#
#     for epoch in range(epochs):
#         feed_dict = {x_placeholder: x_truth,
#                      y_placeholder: y_truth.reshape([-1, 1]),
#                      dropout_placeholder: dropout}
#
#         sess.run(train, feed_dict=feed_dict)
#
#         if epoch % display_step == 0:
#             print("Epoch {}".format(epoch))
#             current_loss = sess.run(loss, feed_dict=feed_dict)
#             print("Loss {}".format(current_loss))
#             print("================")
#
#     print("Training done")
#
#     return sess, x_placeholder, dropout_placeholder
#
#
#


