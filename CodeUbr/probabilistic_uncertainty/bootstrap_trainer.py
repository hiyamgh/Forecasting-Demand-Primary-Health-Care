from scipy.stats import bernoulli
from Code.probabilistic_uncertainty import bootstrap_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


class BootstrapTrainer:

    def __init__(self, X_train, y_train, dropout, learning_rate, epochs, n_heads, display_step=10):

        tf.reset_default_graph()

        self.input_data = tf.placeholder(tf.float64, [None, X_train.shape[1]])
        self.target_data = tf.placeholder(tf.float64, [None, 1])

        self.dropout_placeholder = tf.placeholder(tf.float64)

        # training input & output
        self.X_train = X_train
        self.y_train = y_train

        # other parameters
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.n_heads = n_heads
        self.display_step = display_step

        # This placeholder holds the mask indicating which heads see which samples
        self.mask_placeholder = tf.placeholder(tf.float64, shape=(None, self.n_heads, 1))

        self.heads, self.mean, self.variance = bootstrap_model.bootstrap_model(self.input_data, self.dropout_placeholder,
                                                                               self.n_heads, self.mask_placeholder)

        self.labels = tf.tile(tf.expand_dims(self.target_data, axis=1), [1, self.n_heads, 1])

        # Loss is also only computed on masked heads
        self.loss = tf.nn.l2_loss(self.mask_placeholder * (self.heads - self.labels))

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(init)

        mask_rv = bernoulli(0.5)
        # Since we are not using Mini-Batches computing the mask once suffices
        self.mask = mask_rv.rvs(size=(len(X_train), n_heads, 1))

    def train_model(self):

        for epoch in range(self.epochs):

            # dictionary to feed in
            feed_dict = {self.input_data: self.X_train,
                         self.target_data: self.y_train.reshape([-1, 1]) if self.y_train.ndim == 1 else self.y_train,
                         self.dropout_placeholder: self.dropout,
                         self.mask_placeholder: self.mask}

            # training
            self.sess.run(self.train, feed_dict=feed_dict)

            if epoch % self.display_step == 0:
                print("Epoch {}".format(epoch))
                current_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                print("Loss {}".format(current_loss))
                print("================")

        print("Training done")
        return self.sess, self.input_data, self.dropout_placeholder, self.mask_placeholder


# def bootstrap_training(x_truth, y_truth, dropout, learning_rate, epochs, n_heads, display_step=10):
#     """
#     Generic training of Boostrap Network for 2D data.
#
#     :param x_truth: training samples x
#     :param y_truth: training samples y / label
#     :param dropout:
#     :param learning_rate:
#     :param epochs:
#     :param n_heads: Number of heads for trained Network
#     :param display_step:
#     :return: session, x_placeholder, dropout_placeholder, mask_placeholder
#     """
#     tf.reset_default_graph()
#
#     # for multivariate data
#     x_placeholder = tf.placeholder(tf.float64, [None, x_truth.shape[1]])
#     y_placeholder = tf.placeholder(tf.float64, [None, 1])
#
#     dropout_placeholder = tf.placeholder(tf.float64)
#
#     # This placeholder holds the mask indicating which heads see which samples
#     mask_placeholder = tf.placeholder(tf.float64, shape=(None, n_heads, 1))
#
#     heads, mean, variance = bootstrap_model.bootstrap_model(x_placeholder, dropout_placeholder,
#                                                             n_heads, mask_placeholder)
#     tf.add_to_collection('prediction', mean)
#     tf.add_to_collection('uncertainties', variance)
#     tf.add_to_collection('heads', heads)
#
#     labels = tf.tile(tf.expand_dims(y_placeholder, axis=1), [1, n_heads, 1])
#     # Loss is also only computed on masked heads
#     loss = tf.nn.l2_loss(mask_placeholder * (heads - labels))
#
#     # tf.add_to_collection('loss', loss)
#
#     optimizer = tf.train.AdamOptimizer(learning_rate)
#     train = optimizer.minimize(loss)
#
#     init = tf.global_variables_initializer()
#     sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
#     sess.run(init)
#
#     mask_rv = bernoulli(0.5)
#     # Since we are not using Mini-Batches computing the mask once suffices
#     mask = mask_rv.rvs(size=(len(x_truth), n_heads, 1))
#
#     for epoch in range(epochs):
#         feed_dict = {x_placeholder: x_truth,
#                      y_placeholder: y_truth.reshape([-1, 1]),
#                      dropout_placeholder: dropout,
#                      mask_placeholder: mask}
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
#     return sess, x_placeholder, dropout_placeholder, mask_placeholder