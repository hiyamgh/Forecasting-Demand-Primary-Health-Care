import tensorflow as tf


class DeepEnsembleModel:

    def __init__(self, sizes, num_data, model_scope, learning_rate, max_iter, batch_size, optimizer_name):

        # other network parameters
        self.num_data = num_data
        self.sizes = sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        if optimizer_name not in ['ada_delta', 'ada_grad', 'adam', 'grad_desc']:
            raise ValueError('Oops, you must pick one of the following supported optimizers: {}'.format(['ada_delta', 'ada_grad', 'adam', 'grad_desc']))
        self.optimizer_name = optimizer_name

        # input and target variables
        self.input_data = tf.placeholder(tf.float64, (None, self.num_data))
        self.target_data = tf.placeholder(tf.float64, (None, 1))

        self.network_layers = []
        for i in range(len(self.sizes)):
            # the first layer
            if i == 0:
                curr_dense = [self.num_data, self.sizes[0]]
                self.network_layers.append(curr_dense)

            # the last two layers
            elif i == len(self.sizes) - 2:
                curr_dense = [self.sizes[i - 1], self.sizes[i]]
                self.network_layers.append(curr_dense)
            elif i == len(self.sizes) - 1:
                curr_dense = [self.sizes[i - 2], self.sizes[i]]
                self.network_layers.append(curr_dense)

            # the remaining layers in between
            else:
                curr_dense = [self.sizes[i-1], self.sizes[i]]
                self.network_layers.append(curr_dense)

        layers = self.network_layers

        with tf.variable_scope(model_scope):
            # Densely connect layer variables
            self.weights = []
            self.biases = []

            for i in range(len(self.sizes)):
                self.weights.append(self.weight_variable(model_scope + '_w_fc{}'.format(i), layers[i]))
                self.biases.append(self.bias_variable(model_scope + '_b_fc{}'.format(i), [layers[i][1]]))

        # Network
        # x = input_x
        x = self.input_data
        for i in range(0, len(self.sizes) - 2):
            x = tf.nn.relu(tf.add(tf.matmul(x, self.weights[i]), self.biases[i]))

        # this is the target variable
        self.output_mu = tf.matmul(x, self.weights[-2]) + self.biases[-2]
        self.output_sig = tf.matmul(x, self.weights[-1]) + self.biases[-1]
        self.output_sig_pos = tf.log(1 + tf.exp(self.output_sig)) + 1e-06

        y = self.target_data

        # Negative Log Likelihood(NLL)
        self.loss = tf.reduce_mean(
            0.5 * tf.log(self.output_sig_pos) + 0.5 * tf.div(tf.square(y - self.output_mu), self.output_sig_pos)) + 5

        # Get trainable variables
        self.train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, model_scope)

        # Gradient clipping for preventing nan
        if self.optimizer_name == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'grad_desc':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer_name == 'ada_grad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        else:
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learning_rate)

        self.gvs = self.optimizer.compute_gradients(self.loss, var_list=self.train_vars)
        self.capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in self.gvs]
        self.train_opt = self.optimizer.apply_gradients(self.capped_gvs)

    def weight_variable(self, name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)

    def bias_variable(self, name, shape):
        return tf.get_variable(name, shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float64)




