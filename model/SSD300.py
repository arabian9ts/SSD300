"""
SSD300 is SSD wrapper class.

date: 10/18
author: arabian9ts
"""

import tensorboard as tf
import numpy as np

from model.ssd import *
from matcher import Matcher
from model.computation import *
from model.default_box import *


class SSD300:
    def __init__(self, sess):
        """
        initialize SSD model as SSD300 whose input size is  300x300
        """
        self.sess = sess

        # define input placeholder and initialize ssd instance
        self.input = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
        ssd = SSD()

        # build ssd network => feature-maps and confs and locs tensor is returned
        fmaps, confs, locs = ssd.build(self.input, is_training=True)

        # zip running set of tensor
        self.pred_set = [fmaps, confs, locs]

        # required param from default-box and loss function
        fmap_shapes = [map.get_shape().as_list() for map in fmaps]
        # print('fmap shapes is '+str(fmap_shapes))
        self.dboxes = generate_boxes(fmap_shapes)
        print(len(self.dboxes))

        # required placeholder for loss
        loss, loss_conf, loss_loc, self.pos, self.neg, self.gt_labels, self.gt_boxes = ssd.loss(len(self.dboxes))
        self.train_set = [loss, loss_conf, loss_loc]
        optimizer = tf.train.AdamOptimizer(0.05)
        self.train_step = optimizer.minimize(loss)

        # provides matching method
        self.matcher = Matcher(fmap_shapes, self.dboxes)

    # evaluate loss
    def eval(self, images, actual_data, is_training):
        if not is_training:
            feature_maps, pred_confs, pred_locs = self.sess.run(self.pred_set, feed_dict={self.input: images})
            return pred_confs, pred_locs

        # ================ RESET / EVAL ================ #
        actual_labels = []
        actual_locs = []
        positives = []
        negatives = []
        ex_gt_labels = []
        ex_gt_boxes = []
        # ===================== END ===================== #

        # call prepare_loss per image
        # because matching method works with only one image
        def prepare_loss(pred_confs, pred_locs, actual_labels, actual_locs):
            pos_list, neg_list, t_gtl, t_gtb = self.matcher.matching(pred_confs, pred_locs, actual_labels, actual_locs)
            positives.append(pos_list)
            negatives.append(neg_list)
            ex_gt_labels.append(t_gtl)
            ex_gt_boxes.append(t_gtb)


        for i in range(len(images)):
            # extract ground truth info
            for obj in actual_data[i]:
                loc = obj[:4]
                label = np.argmax(obj[4:])

                # transform location for ssd-training
                loc = corner2center(loc)

                actual_locs.append(loc)
                actual_labels.append(label)

            prepare_loss(pred_confs, pred_locs, actual_labels, actual_locs)
                
        batch_loss, batch_conf, batch_loc = \
        self.sess.run(self.train_set, \
        feed_dict={self.input: images, self.pos: positives, self.neg: negatives, self.gt_labels: ex_gt_labels, self.gt_boxes: ex_gt_boxes})

        self.sess.run(self.train_step, \
        feed_dict={self.input: images, self.pos: positives, self.neg: negatives, self.gt_labels: ex_gt_labels, self.gt_boxes: ex_gt_boxes})

        return pred_confs, pred_locs, batch_loc, batch_conf, batch_loss