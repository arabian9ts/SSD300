"""
training script

date: 10/4
author: arabian9ts
"""

import tensorboard as tf
import numpy as np

from model.ssd import *
from matcher import Matcher
from model.default_box import *



input = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
ssd = SSD()
fmaps = ssd.build(input, is_training=True)
fmap_shapes = [map.get_shape().as_list() for map in fmaps]
print('fmap shapes is '+str(fmap_shapes))
dboxes = generate_boxes(fmap_shapes)
print(len(boxes))

loss = ssd.loss(ssd.pred_confs, ssd.pred_locs, len(dboxes))

matcher = Matcher(fmap_shapes, dboxes)

actual_labels = np.array([1, 2])
actual_locs = np.array([[0.5, 0.5, 1, 1], [4, 4, 4, 4]])

print(matcher.matching(ssd.pred_confs, ssd.pred_locs, actual_labels, actual_locs))