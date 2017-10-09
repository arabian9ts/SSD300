"""
training script

date: 10/4
author: arabian9ts
"""

import tensorboard as tf
import numpy as np

from model.ssd import *
from util.util import *
from matcher import Matcher
from model.default_box import *


# ====================== Training Parameters ====================== #
BATCH_SIZE = 1
positives = []
negatives = []
# ============================== END ============================== #


# ====================== SSD Setup ====================== #
input = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
ssd = SSD()
fmaps, confs, locs = ssd.build(input, is_training=True)
train_set = [fmaps, confs, locs]
fmap_shapes = [map.get_shape().as_list() for map in fmaps]
print('fmap shapes is '+str(fmap_shapes))
dboxes = generate_boxes(fmap_shapes)
print(len(boxes))

loss, pos, neg, gt_labels, gt_boxes = ssd.loss(len(dboxes))

matcher = Matcher(fmap_shapes, dboxes)

actual_labels = np.array([1, 2])
actual_locs = np.array([[0.5, 0.5, 1, 1], [4, 4, 4, 4]])

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    image = load_image('./test.jpg')
    image = image.reshape((1, 300, 300, 3))

    def prepare_loss(pred_confs, pred_locs, actual_labels, actual_locs):
        global positives, negatives
        matches, pos_list, neg_list = matcher.matching(pred_confs, pred_locs, actual_labels, actual_locs)
        positives.append(pos_list)
        negatives.append(neg_list)


    feature_maps, pred_confs, pred_locs = sess.run(train_set, feed_dict={input: image})
    print(len(pred_confs))
    prepare_loss(pred_confs, pred_locs, actual_labels, actual_locs)
    # sess.run(loss, feed_dict={input: image, pos: positives, neg: negatives, gt_labels: actual_labels, gt_boxes: actual_locs})