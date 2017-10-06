"""
Enumeration class of tensorflow-activation methods

date: 9/18
author: arabian9ts
"""

import enum
import tensorflow as tf

class Activation(enum.Enum):
    identity = tf.identity
    relu = tf.nn.relu
    softmax = tf.nn.softmax