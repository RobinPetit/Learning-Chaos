# -*- coding: utf-8 -*-
# dddqn.py : Double dueling deep Q-network
# author : Robin Petit, Stanislas Gueniffey, Cedric Simar, Antoine Passemiers

from dqn import DQN
from tf_decorator import *
from parameters import Parameters

import tensorflow as tf
from tensorflow.python.ops import array_ops as tf_array_ops


class DDDQN(DQN):

    def __init__(self, state):
        DQN.__init__(self, state)

    @define_scope
    def q_values(self):
        """
        References
        ----------
        1)  https://arxiv.org/pdf/1511.06581.pdf
            Dueling Network Architectures for Deep Reinforcement Learning
            Wang et al.
        2)  https://github.com/awjuliani/DeepRL-Agents/blob/master/Double-Dueling-DQN.ipynb
            Simple Reinforcement Learning with Tensorflow Part 4: Deep Q-Networks and Beyond
        """
        
        """
        [Article] The input to the neural network (the state) consists of an 84 x 84 x 4 image 
        produced by the preprocessing map "phi"
        """
        # reshape input to 4d tensor [batch, height, width, channels]
        input = tf.reshape(self.state, [-1, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.AGENT_HISTORY_LENGTH])

        # convolutional layer 1
        """
        [Article] The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the
        input image and applies a rectifier nonlinearity
        """
        W_conv1 = self.weight_variable([8, 8, Parameters.AGENT_HISTORY_LENGTH, 32])
        b_conv1 = self.bias_variable([32])
        conv1 = tf.nn.conv2d(input, W_conv1, strides=[1, 4, 4, 1], padding='VALID') # would 'SAME' also work ?
        h_conv1 = tf.nn.relu(conv1 + b_conv1)
        
        # output of conv 1 is of shape [-1 x 20 x 20 x 32]
        
        # convolutional layer 2
        """
        [Article] The second hidden layer convolves 64 filters of 4 x 4 with stride 2, 
        again followed by a rectifier nonlinearity
        """
        W_conv2 = self.weight_variable([4, 4, 32, 64])
        b_conv2 = self.bias_variable([64])
        conv2 = tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 2, 2, 1], padding='VALID')
        h_conv2 = tf.nn.relu(conv2 + b_conv2)

        # output of conv 2 is of shape [-1 x 9 x 9 x 64]

        # convolutional layer 3
        """
        [Article] This is followed by a third convolutional layer that convolves 
        64 filters of 3 x 3 with stride 1 followed by a rectifier
        """
        W_conv3 = self.weight_variable([3, 3, 64, 64])
        b_conv3 = self.bias_variable([64])
        conv3 = tf.nn.conv2d(h_conv2, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
        h_conv3 = tf.nn.relu(conv3 + b_conv3)

        # output of conv 3 is of shape [-1 x 7 x 7 x 64]
        # this output is then split into two separate streams:
        # avantage-per-action (apa) stream and state-value (sv) stream
        apa_stream, sv_stream = tf.split(h_conv3, 2, axis=3)

        # Flatten streams individually
        apa_stream = tf.contrib.layers.flatten(apa_stream)
        sv_stream = tf.contrib.layers.flatten(sv_stream)

        # Add one fully connected linear layer per stream
        W_apa = self.weight_variable([1568, Parameters.ACTION_SPACE], method="xavier")
        W_sv = self.weight_variable([1568, 1], method="xavier")
        fc_apa = tf.matmul(apa_stream, W_apa)
        fc_sv = tf.matmul(sv_stream, W_sv)
        
        # alpha = parameters of avantage-per-action stream
        # beta = parameters of state-value stream
        # theta = parameters of the rest of the network
        # Q(s,a;theta,alpha,beta) = V(s;theta,beta) + ( A(s,a;theta,alpha) - 1/|A| * Sum_(a') A(s,a';theta,alpha) ).
        predicted_q_values = fc_sv + tf.subtract(fc_apa, tf.reduce_mean(fc_apa, axis=1, keep_dims=True))
        

        # saving learning parameters and layers output to access them directly if needed

        self.learning_parameters["W_conv1"], self.learning_parameters["b_conv1"] = W_conv1, b_conv1
        self.layers["conv1"], self.layers["h_conv1"] = conv1, h_conv1

        self.learning_parameters["W_conv2"], self.learning_parameters["b_conv2"] = W_conv2, b_conv2
        self.layers["conv2"], self.layers["h_conv2"] = conv2, h_conv2

        self.learning_parameters["W_conv3"], self.learning_parameters["b_conv3"] = W_conv3, b_conv3
        self.layers["conv3"], self.layers["h_conv3"] = conv3, h_conv3
        
        self.learning_parameters["W_fc_apa"] = W_apa
        self.layers["fc_apa"] = fc_apa
        
        self.learning_parameters["W_fc_sv"] = W_sv
        self.layers["fc_sv"] = fc_sv

        return(predicted_q_values)
