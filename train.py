"""
training script

date: 10/4
author: arabian9ts
"""

# escape matplotlib error
import matplotlib
matplotlib.use('Agg')

# escape tensorflow warning
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import datetime
import tensorboard as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from model.ssd import *
from util.util import *
from matcher import Matcher
from model.computation import *
from model.default_box import *


# ====================== Training Parameters ====================== #
BATCH_SIZE = 20
EPOCH = 30
EPOCH_LOSSES = []
# ============================== END ============================== #


# ====================== SSD Setup ====================== #
class SSD300:
    def __init__(self):
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

        feature_maps, pred_confs, pred_locs = sess.run(self.pred_set, feed_dict={self.input: images})

        for i in range(len(images)):
            # extract ground truth info
            for obj in actual_data[i]:
                loc = obj[:4]
                label = np.argmax(obj[4:])

                # transform location for ssd-training
                loc = corner2center(swap_width_height(loc))

                actual_locs.append(loc)
                actual_labels.append(label)

            prepare_loss(pred_confs, pred_locs, actual_labels, actual_locs)
                
        batch_loss, batch_conf, batch_loc = \
        sess.run(self.train_set, \
        feed_dict={self.input: images, self.pos: positives, self.neg: negatives, self.gt_labels: ex_gt_labels, self.gt_boxes: ex_gt_boxes})

        if is_training:
            sess.run(self.train_step, \
            feed_dict={self.input: images, self.pos: positives, self.neg: negatives, self.gt_labels: ex_gt_labels, self.gt_boxes: ex_gt_boxes})

        return pred_confs, pred_locs, batch_loc, batch_conf, batch_loss



if __name__ == '__main__':
    ssd = SSD300()

    # load pickle data set annotation
    with open('VOC2007.pkl', 'rb') as f:
        data = pickle.load(f)
        keys = sorted(data.keys())
        slicer = int(len(keys) * 0.8)
        train_keys = keys[:slicer]
        test_keys = keys[slicer:]
        BATCH = int(len(train_keys) / BATCH_SIZE)

    def next_batch(is_training):
        mini_batch = []
        actual_data = []
        if is_training:
            indicies = np.random.choice(len(train_keys), BATCH_SIZE)
        else:
            indicies = np.random.choice(len(test_keys), BATCH_SIZE)

        for idx in indicies:
            # make images mini batch
            if is_training:
                img = load_image('voc2007/'+train_keys[idx])
                actual_data.append(data[train_keys[idx]])
            else:
                img = load_image('voc2007/'+test_keys[idx])
                actual_data.append(data[test_keys[idx]])

            img = img.reshape((300, 300, 3))
            mini_batch.append(img)

        return mini_batch, actual_data

    # tensorflow session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print('\nSTART LEARNING')
        print('==================== '+str(datetime.datetime.now())+' ====================')

        for ep in range(EPOCH):
            BATCH_LOSSES = []
            for ba in range(BATCH):
                minibatch, actual_data = next_batch(is_training=True)
                _, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, True)
                BATCH_LOSSES.append(batch_loss)

                print('\n********** BATCH LOSS **********')
                print('       LOC LOSS: '+str(batch_loc))
                print('       CONF LOSS: '+str(batch_conf))
                print('       TOTAL LOSS: '+str(batch_loss))
                print('========== BATCH: '+str(ba+1)+' END ==========')
            EPOCH_LOSSES.append(np.mean(BATCH_LOSSES))
            print('\n*** AVERAGE: '+str(EPOCH_LOSSES[-1])+' ***')

            print('\n*** TEST ***')
            minibatch, actual_data = next_batch(is_training=False)
            _, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, False)
            print('LOC LOSS: '+str(batch_loc))
            print('CONF LOSS: '+str(batch_conf))
            print('TOTAL LOSS: '+str(batch_loss))
            print('\n========== EPOCH: '+str(ep+1)+' END ==========')
            
        print('\nEND LEARNING')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(np.array(range(EPOCH)), EPOCH_LOSSES)
        plt.grid()
        plt.savefig("loss.png")
        plt.show()

        print('==================== '+str(datetime.datetime.now())+' ====================')