"""
implementation of vgg network with TensorFlow

date: 9/17
author: arabian9ts
"""

import tensorflow as tf
import numpy as np

from model.tf_material import *

class VGG16:
    """
    This is presented by VGG team in oxford Univ.
    """

    def __init__(self):
        pass

    def build(self, input, is_training=True):
        """
        input is the placeholder of tensorflow
        build func assembles vgg16 network

        Args:
            input: images batch
            is_training: is this training?
        Returns:
            last output of this network sequence.
        """

        # flag: is_training? for tensorflow-graph
        self.train_phase = tf.constant(is_training) if is_training else None

        self.conv1_1 = convolution(input, 'conv1_1')
        self.conv1_2 = convolution(self.conv1_1, 'conv1_2')
        self.pool1 = pooling(self.conv1_2, 'pool1')

        self.conv2_1 = convolution(self.pool1, 'conv2_1')
        self.conv2_2 = convolution(self.conv2_1, 'conv2_2')
        self.pool2 = pooling(self.conv2_2, 'pool2')

        self.conv3_1 = convolution(self.pool2, 'conv3_1')
        self.conv3_2 = convolution(self.conv3_1, 'conv3_2')
        self.conv3_3 = convolution(self.conv3_2, 'conv3_3')
        self.pool3 = pooling(self.conv3_3, 'pool3')

        self.conv4_1 = convolution(self.pool3, 'conv4_1')
        self.conv4_2 = convolution(self.conv4_1, 'conv4_2')
        self.conv4_3 = convolution(self.conv4_2, 'conv4_3')
        self.pool4 = pooling(self.conv4_3, 'pool4')

        self.conv5_1 = convolution(self.pool4, 'conv5_1')
        self.conv5_2 = convolution(self.conv5_1, 'conv5_2')
        # self.conv5_3 = convolution(self.conv5_2, 'conv5_3')
        # self.pool5 = self.pooling(self.conv5_3, 'pool5')

        # self.fc6 = self.fully_connection(self.pool5, Activation.relu, 'fc6')
        # self.fc7 = self.fully_connection(self.fc6, Activation.relu, 'fc7')
        # self.fc8 = self.fully_connection(self.fc7, Activation.softmax, 'fc8')

        self.prob = self.conv5_2

        return self.prob
