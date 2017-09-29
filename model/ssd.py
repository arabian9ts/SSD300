"""
implementation of ssd network with TensorFlow

SSD model has own network, loss.
Because of SSD algo, use default box, so Box-func rewuired.

** Default Box **
Each pixel in feature-map extracted from original-images have 3 boxes(defalut boxes).
They are slaced(e.g. to sqrt(2) : 1/sqrt(2) or transposed).
Default Boxes have own position params.

** Loss function **
SSD uses specific loss using locational_loss and confidential_loss.


Note:
    The number of Default Box:
        [ 3, 6, 6, 6, 6, 6, ]
        <-> from [ conv5_3,  ]

    Default Box params:
        loc = ( center_x, center_y, box_height, box_width )

    Loss function:
        Loss = (Loss_conf + a*Loss_loc) / N
        N is the number of matched default boxes.
        If N==0 -> Loss = 0

    Box scale(meaning size):
        s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0), m = 6

    Box scale ratio(height and width scaling):
        Recommended params are [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0].
        However, if output is conv5_2, use [1.0, 0.5, 2.0].
        I use these.
        Actually, box_height = s_k/sqrt(ratio) and box_width = s_k*sqrt(ratio)
        If ratio is 1, additional box whose scale is s'_k = sqrt(s_k*s_{k+1})


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
        base = super().build(input, is_training)
        return base




input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32)
fmap = SSD().build(input, is_training=True)