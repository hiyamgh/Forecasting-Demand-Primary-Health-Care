from Code.probabilistic_uncertainty import combined_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class CombinedTrainer:

    def __init__(self, X_train, y_train, dropout, learning_rate, epochs, display_step=10):

        # training input & output
        self.X_train = X_train
        self.y_train = y_train

        # other variables
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.display_step = display_step

        self.input_data = tf.placeholder(tf.float64, [None, self.X_train.shape[1]])
        self.target_data = tf.placeholder(tf.float64, [None, 1])

        self.dropout_placeholder = tf.placeholder(tf.float64)

        self.prediction, self.log_variance = combined_model.combined_model(self.input_data, self.dropout_placeholder)
        self.loss = tf.reduce_sum(0.5 * tf.exp(-1 * self.log_variance) * tf.square(tf.abs(self.target_data - self.prediction))
                         + 0.5 * self.log_variance)

        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train = self.optimizer.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
        self.sess.run(init)

    def train_model(self):
        for epoch in range(self.epochs):

            # dictionary to feed in
            feed_dict = {self.input_data: self.X_train,
                         self.target_data: self.y_train.reshape([-1, 1]) if self.y_train.ndim == 1 else self.y_train,
                         self.dropout_placeholder: self.dropout}

            # train
            self.sess.run(self.train, feed_dict=feed_dict)

            if epoch % self.display_step == 0:
                print("Epoch {}".format(epoch))
                current_loss = self.sess.run(self.loss, feed_dict=feed_dict)
                print("Loss {}".format(current_loss))
                print("================")

        print("Training done")

        return self.sess, self.input_data, self.dropout_placeholder


# def combined_training(x_truth, y_truth, dropout, learning_rate, epochs, display_step=10):
#     """
#     Generic training of a Combined (uncertainty) network for 2D data.
#
#     :param x_truth: training samples x
#     :param y_truth: training samples y / label
#     :param dropout:
#     :param learning_rate:
#     :param epochs:
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
#
#     prediction, log_variance = combined_model.combined_model(x_placeholder, dropout_placeholder)
#
#     tf.add_to_collection("prediction", prediction)
#     tf.add_to_collection("log_variance", log_variance)
#
#     # is this the NLL :) ??
#     loss = tf.reduce_sum(0.5 * tf.exp(-1 * log_variance) * tf.square(tf.abs(y_placeholder - prediction))
#                          + 0.5 * log_variance)
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
#     for epoch in range(epochs):
#         # print('epochhhhh: {}'.format(epoch))
#         feed_dict = {x_placeholder: x_truth,
#                      y_placeholder: y_truth.reshape([-1, 1]),
#                      dropout_placeholder: dropout}
#
#         sess.run(train, feed_dict=feed_dict)
#         # print("Epoch {}".format(epoch))
#         # current_loss = sess.run(loss, feed_dict=feed_dict)
#         # print("Loss {}".format(current_loss))
#         # print("================")
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
