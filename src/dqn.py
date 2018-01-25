# -*- coding: utf-8 -*-
# dqn.py : Deep Q-Networks
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from tf_decorator import *
from parameters import Parameters

import tensorflow as tf
from tensorflow.python.ops import array_ops as tf_array_ops


class DQN:

    def __init__(self, state, action_space):

        # receiving state placeholder
        self.state = state

        # creating importance-sampling weights placeholder
        self.i_s_weights = tf.placeholder(
            tf.float32, [None], name="i_s_weights")

        # set action space size
        self.action_space = action_space

        # weights and biases dictionary
        self.learning_parameters = {}
        self.layers = {}

        # initialize tensorflow graph
        self.q_values
        self.optimize
        self.error

        # Initialize input placeholder to assign values to weights and biases
        # Used when we transfert them from the training network to the target network
        # or when we load DQN parameters previously saved in a file
        with tf.variable_scope("test"):

            self.l_param_input = {}
            self.assign_operator = {}
            for variable_name in self.learning_parameters.keys():
                self.l_param_input[variable_name] = tf.placeholder(
                    tf.float32,
                    self.learning_parameters[variable_name].get_shape().as_list(),
                    name=variable_name)

                try:  # If mutable tensor (Variable)
                    self.assign_operator[variable_name] = self.learning_parameters[variable_name].assign(
                        self.l_param_input[variable_name])
                except AttributeError as e:
                    print(e)

    @define_scope
    def q_values(self):
        """
        [Article] The input to the neural network (the state) consists of an 84 x 84 x 4 image
        produced by the preprocessing map "phi"
        """
        # reshape input to 4d tensor [batch, height, width, channels]
        input = tf.reshape(self.state,
                   [-1,
                    Parameters.IMAGE_HEIGHT,
                    Parameters.IMAGE_WIDTH,
                    Parameters.AGENT_HISTORY_LENGTH])

        # convolutional layer 1
        """
        [Article] The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the
        input image and applies a rectifier nonlinearity
        """
        W_conv1 = self.weight_variable(
            [8, 8, Parameters.AGENT_HISTORY_LENGTH, 32])
        b_conv1 = self.bias_variable([32])
        conv1 = tf.nn.conv2d(
            input,
            W_conv1,
            strides=[1, 4, 4, 1],
            padding='VALID')  # would 'SAME' also work ?
        h_conv1 = tf.nn.relu(conv1 + b_conv1)

        # output of conv 1 is of shape [-1 x 20 x 20 x 32]

        # convolutional layer 2
        """
        [Article] The second hidden layer convolves 64 filters of 4 x 4 with stride 2,
        again followed by a rectifier nonlinearity
        """
        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])
        conv2 = tf.nn.conv2d(
            h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID')
        h_conv2 = tf.nn.relu(conv2 + b_conv2)

        # output of conv 2 is of shape [-1 x 9 x 9 x 64]

        # convolutional layer 3
        """
        [Article] This is followed by a third convolutional layer that convolves
        64 filters of 3 x 3 with stride 1 followed by a rectifier
        """
        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
        conv3 = tf.nn.conv2d(
            h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        h_conv3 = tf.nn.relu(conv3 + b_conv3)

        # output of conv 3 is of shape [-1 x 7 x 7 x 64]

        h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64])

        # fully connected layer 1
        W_fc1 = self.weight_variable([7 * 7 * 64, 512])
        b_fc1 = self.bias_variable([512])
        fc1 = tf.matmul(h_conv3_flat, W_fc1)
        self.h_fc1 = h_fc1 = tf.nn.relu(fc1 + b_fc1)

        # fully connected layer 2 (output layer)
        W_fc2 = self.weight_variable([512, self.action_space])
        b_fc2 = self.bias_variable([self.action_space])
        fc2 = tf.matmul(h_fc1, W_fc2)

        # network output is of shape (1, self.action_space)
        """
        [Article] We use an architecture in which there is a separate output unit
        for each possible action [ = one-hot encoding ], and only the state representation
        is an input to the neural network. The outputs correspond to the predicted Q-values of
        the individual actions for the input state.
        """
        predicted_q_values = fc2 + b_fc2

        # saving learning parameters and layers output to access them directly
        # if needed

        self.learning_parameters["W_conv1"], self.learning_parameters["b_conv1"] = W_conv1, b_conv1
        self.layers["conv1"], self.layers["h_conv1"] = conv1, h_conv1

        self.learning_parameters["W_conv2"], self.learning_parameters["b_conv2"] = W_conv2, b_conv2
        self.layers["conv2"], self.layers["h_conv2"] = conv2, h_conv2

        self.learning_parameters["W_conv3"], self.learning_parameters["b_conv3"] = W_conv3, b_conv3
        self.layers["conv3"], self.layers["h_conv3"], self.layers["h_conv3_flat"] = conv3, h_conv3, h_conv3_flat

        self.learning_parameters["W_fc1"], self.learning_parameters["b_fc1"] = W_fc1, b_fc1
        self.layers["fc1"], self.layers["h_fc1"] = fc1, h_fc1

        self.learning_parameters["W_fc2"], self.learning_parameters["b_fc2"] = W_fc2, b_fc2
        self.layers["fc2"], self.layers["h_fc2"] = fc2, predicted_q_values

        return(predicted_q_values)

    @define_scope
    def smartest_action(self):

        highest_q_value = tf.argmax(self.q_values, axis=1)
        return(highest_q_value)

    @define_scope
    def optimize(self):

        self.optimizer = tf.train.RMSPropOptimizer(
            learning_rate=Parameters.LEARNING_RATE,
            momentum=Parameters.GRADIENT_MOMENTUM,
            epsilon=Parameters.MIN_SQUARED_GRADIENT)

        return(self.optimizer.minimize(self.importance_weighted_error))

    @define_scope
    def errors(self):
        """ Return the clipped errors """
        # placeholders for the target network q values and the action
        self.target_q = tf.placeholder(tf.float32, [None], name="target_q")
        self.action = tf.placeholder(tf.int64, [None], name="action")

        # convert the action to one-hot representation in order to compute the
        # error
        action_one_hot = tf.one_hot(
            self.action,
            self.action_space,
            on_value=1,
            off_value=0,
            name="action_one_hot")

        self.q_acted = tf.reduce_sum(
            self.q_values *
            tf.cast(
                action_one_hot,
                tf.float32),
            axis=1,
            name="q_acted")

        self.delta = self.target_q - self.q_acted

        """
        [Article] We also found it helpful to clip the error term from the update r + gamma max_d Q(s', a', theta-)
        to be between -1 and 1. Because the absolute value loss function |x| has a derivative of -1
        for all negative values of x and a derivative of 1 for all positive values of x,
        clipping the squared error to be between -1 and 1 corresponds to using an absolute value
        loss function for errors outside of the (-1,1) interval. This form of error clipping further
        improved the stability of the algorithm.

        It is called the Huber loss and because the name is so cool, we have to implement it
        With d = 1 (we could also try with d = 2) (d <> self.delta)
        x =  0.5 * x^2                  if |x| <= d
        x =  0.5 * d^2 + d * (|x| - d)  if |x| > d
        """
        self.clipped_error = tf_array_ops.where(tf.abs(self.delta) < 1.0,
                                                tf.square(self.delta) * 0.5,
                                                tf.abs(self.delta) - 0.5)
        return(self.clipped_error)

    @define_scope
    def error(self):
        """ Return the mean (clipped) error """
        self.mean_error = tf.reduce_mean(self.errors, name="mean_error")
        return(self.mean_error)

    @define_scope
    def importance_weighted_error(self):
        """ Return the importance-weighted error for prioritized memory updates """
        weighted_errors = self.i_s_weights * self.errors
        self.mean_error = tf.reduce_mean(weighted_errors, name="mean_error")
        return(self.mean_error)

    def weight_variable(self, shape, method="normal"):
        """
        Initialize weight variable randomly using one of the two following methods:
            - A truncated normal distribution of mean = 0 and standard deviation of 0.02
            - A xavier initialization, where var(W) = 1 / n_in
        """
        assert(method in ["normal", "xavier"])
        if method == "normal":
            weight_var = tf.truncated_normal(shape, mean=0, stddev=0.02)
        else:
            xavier_initializer = tf.contrib.layers.xavier_initializer()
            weight_var = xavier_initializer(list(shape))
        return(tf.Variable(weight_var))

    def bias_variable(self, shape):
        """
        Initialize bias variables of a specific shape using a constant
        """
        bias_var = tf.constant(0.1, shape=shape)
        return(tf.Variable(bias_var))

    def get_value(self, var_name, tf_session):
        """
        Return the value of the tf variable named [var_name] if it exists, None otherwise
        """

        if var_name in self.learning_parameters:

            value = tf_session.run(self.learning_parameters[var_name])

        elif var_name in self.layers:

            value = tf_session.run(self.layers[var_name])

        else:
            print("Unknown DQN variable: " + var_name)
            assert(0)  # <3

        return(value)

    def set_value(self, var_name, new_value, tf_session):
        """
        Set the value of the tf variable [var_name] to [new_value]
        """

        if(var_name in self.assign_operator):

            tf_session.run(
                self.assign_operator[var_name], {
                    self.l_param_input[var_name]: new_value})
        else:
            print("Thou shall only assign learning parameters!")
