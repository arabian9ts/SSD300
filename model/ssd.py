"""
implementation of ssd network with TensorFlow

SSD model has own network, loss.
Because of SSD algo, use default box, so Box-func required.

** Default Box **
Each pixel in feature-map extracted from original-images have 3 boxes(defalut boxes).
They are slaced(e.g. to sqrt(2) : 1/sqrt(2) or transposed).
Default Boxes have own position params.

** Loss function **
SSD uses specific loss using locational_loss and confidential_loss.


Note:
    The number of Default Box:
        [ 3, 6, 6, 6, 6, 6, ]
        <-> from [ base, conv7, conv8_2, conv9_2, conv10_2, conv11_2 ]

    Default Box params:
        loc = ( center_x, center_y, box_height, box_width )

    Loss function:
        Loss = (Loss_conf + a*Loss_loc) / N
        N is the number of matched default boxes.
        If N==0 -> Loss = 0

    Box scale(meaning size scaling):
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
from default_box import *
from tf_material import *

class SSD(VGG16):

    def __init__(self):
        super()

    def build(self, input, is_training=True):
        """
        SSD network by use of VGG16.
        This assembles SSD network by extending VGG16 class.

        Args:
            input: images batch
            is_training: is this training?

        Returns:
            feature maps
        """

        # assenble base network
        self.base = super().build(input, is_training)
        
        self.conv6 = convolution(self.base, 'conv6')
        self.conv7 = convolution(self.conv6, 'conv7')

        self.conv8_1 = convolution(self.conv7, 'conv8_1')
        self.conv8_2 = convolution(self.conv8_1, 'conv8_2', ksize=3, stride=2)

        self.conv9_1 = convolution(self.conv8_2, 'conv9_1')
        self.conv9_2 = convolution(self.conv9_1, 'conv9_2', ksize=3, stride=2)

        self.conv10_1 = convolution(self.conv9_2, 'conv10_1')
        self.conv10_2 = convolution(self.conv10_1, 'conv10_2', ksize=3, stride=2)

        self.conv11_1 = convolution(self.conv10_2, 'conv11_1')
        self.conv11_2 = convolution(self.conv11_1, 'conv11_2', stride=3)

        print('================== Feature Map Below ==================')

        self.feature_maps = []
        # extra feature maps
        self.feature_maps.append(convolution(self.base, 'map1'))
        self.feature_maps.append(convolution(self.conv7, 'map2'))
        self.feature_maps.append(convolution(self.conv8_2, 'map3'))
        self.feature_maps.append(convolution(self.conv9_2, 'map4'))
        self.feature_maps.append(convolution(self.conv10_2, 'map5'))
        self.feature_maps.append(convolution(self.conv11_2, 'map6'))

        pred = []
        for i, fmap in zip(range(len(feature_maps)), feature_maps):
            output_size = fmap.get_shape().as_list()
            height = output_size[1]
            width = output_size[2]
            pred.append(tf.reshape(fmap, [-1, width*height*boxes[i], classes+4]))

        concatenated = tf.concat(pred, axis=1)
        
        self.pred_confs = concatenated[:,:,:classes]
        self.pred_locs = concatenated[:,:,classes:]

        print('concatenated: '+str(concatenated))
        print('confs: '+str(pred_confs.get_shape().as_list()))
        print('locs: '+str(pred_locs.get_shape().as_list()))
                
        self.default_boxes = generate_boxes([map.get_shape().as_list() for map in feature_maps])
        print(len(self.default_boxes))

        return pred


    def loss():
        """
        loss func defined as Loss = (Loss_conf + a*Loss_loc) / N
        In here, compute confidence loss and location loss,
        finally, total loss.

        Returns: total loss per batch
        """

        return



input = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
fmap = SSD().build(input, is_training=True)