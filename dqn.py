
import tensorflow as tf
from tf_decorator import *
from parameters import Parameters


class DQN:

    def __init__(self, image):

        # receiving image placeholder
        self.image = image

        # initialize tensorflow graph
        self.prediction
        self.train
        self.error


    @define_scope
    def prediction(self):
        
        """
        [Article] The input to the neural network consists of an 84 x 84 x 4 image 
        produced by the preprocessing map ϕ
        """
        # reshape input to 4d tensor [batch, height, width, channels]
        input = tf.reshape(self.image, [-1, Parameters.IMAGE_HEIGHT, Parameters.IMAGE_WIDTH, Parameters.M_RECENT_FRAMES])

        # convolutional layer 1
        """
        [Article] The first hidden layer convolves 32 filters of 8 x 8 with stride 4 with the
        input image and applies a rectifier nonlinearity
        """
        W_conv1 = self.weight_variable([8, 8, Parameters.M_RECENT_FRAMES, 32])
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
       
        h_conv3_flat = tf.reshape(h_conv3, [-1, 7 * 7 * 64])

        # fully connected layer 1
        W_fc1 = self.weight_variable([7 * 7 * 64, 512])
        b_fc1 = self.bias_variable([512])
        fc1 = tf.matmul(h_conv3_flat, W_fc1)
        h_fc1 = tf.nn.relu(fc1 + b_fc1)

        # fully connected layer 2 (output layer)
        W_fc2 = self.weight_variable([512, Parameters.ACTION_SPACE])
        b_fc2 = self.bias_variable([Parameters.ACTION_SPACE])
        fc2 = tf.matmul(h_fc1, W_fc2)

        net_output = fc2 + b_fc2

        # network output is of shape (1, Parameters.ACTION_SPACE)
        
        return(net_output)
    

    @define_scope
    def train(self):
        print("Not available yet")
        """ (MNIST) NOT YET UPDATED FOR THE PROJECT """
        # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.prediction))
        # optimizer = tf.train.AdamOptimizer(1e-4)
        # return(optimizer.minimize(cross_entropy))


    @define_scope
    def error(self):
        print("Not available yet")
        """ (MNIST) NOT YET UPDATED FOR THE PROJECT """
        # errors = tf.not_equal(tf.argmax(self.label, 1), tf.argmax(self.prediction, 1))
        # return(tf.reduce_mean(tf.cast(errors, tf.float32)))


    def weight_variable(self, shape):
        """ (MNIST) NOT YET UPDATED FOR THE PROJECT """
        weight_var = tf.truncated_normal(shape, stddev=0.1)
        return(tf.Variable(weight_var))


    def bias_variable(self, shape):
        """ (MNIST) NOT YET UPDATED FOR THE PROJECT """
        bias_var = tf.constant(0.1, shape=shape)
        return(bias_var)

