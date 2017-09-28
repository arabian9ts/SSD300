"""
implementation of ssd network with TensorFlow

date: 9/28
author: arabian9ts
"""

import tensorflow as tf
import numpy as np

from base import VGG16

class SSD(VGG16):

    def __init__(self):
        super()

    def build(self, input, is_training=True):
        super().build(input, is_training=True)




input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
fmap = SSD().build(input, is_training=True)