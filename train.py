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
BATCH_SIZE = 2
BATCH = int(4592 / BATCH_SIZE)
EPOCH = 10
MINIBATCH = []
EPOCH_LOSSES = []
BATCH_LOSSES = []
# ============================== END ============================== #


# ====================== SSD Setup ====================== #

# define input placeholder and initialize ssd instance
input = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
ssd = SSD()

# build ssd network => feature-maps and confs and locs tensor is returned
fmaps, confs, locs = ssd.build(input, is_training=True)

# zip running set of tensor
train_set = [fmaps, confs, locs]

# required param from default-box and loss function
fmap_shapes = [map.get_shape().as_list() for map in fmaps]
# print('fmap shapes is '+str(fmap_shapes))
dboxes = generate_boxes(fmap_shapes)
print(len(dboxes))

# required placeholder for loss
loss, pos, neg, gt_labels, gt_boxes = ssd.loss(len(dboxes))
optimizer = tf.train.AdamOptimizer(0.05)
train_step = optimizer.minimize(loss)

# provides matching method
matcher = Matcher(fmap_shapes, dboxes)

# load pickle data set annotation
with open('VOC2007.pkl', 'rb') as f:
    data = pickle.load(f)
    keys = sorted(data.keys())

# tensorflow session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    print('\nSTART LEARNING')
    print('==================== '+str(datetime.datetime.now())+' ====================')

    for ep in range(EPOCH):
        BATCH_LOSSES = []
        for ba in range(BATCH):

            # ================ RESET / BATCH ================ #
            MINIBATCH = []
            actual_labels = []
            actual_locs = []
            positives = []
            negatives = []
            ex_gt_labels = []
            ex_gt_boxes = []
            indicies = np.random.choice(len(keys), BATCH_SIZE)
            # ===================== END ===================== #

            # call prepare_loss per image
            # because matching method works with only an image
            def prepare_loss(pred_confs, pred_locs, actual_labels, actual_locs):
                global positives, negatives, ex_gt_labels, ex_gt_boxes
                pos_list, neg_list, t_gtl, t_gtb = matcher.matching(pred_confs, pred_locs, actual_labels, actual_locs)
                positives.append(pos_list)
                negatives.append(neg_list)
                ex_gt_labels.append(t_gtl)
                ex_gt_boxes.append(t_gtb)

            
            for idx in indicies:
                # make images mini batch
                img = load_image('voc2007/'+keys[idx])
                img = img.reshape((300, 300, 3))
                MINIBATCH.append(img)

            feature_maps, pred_confs, pred_locs = sess.run(train_set, feed_dict={input: MINIBATCH})

            for idx in indicies:
                 # extract ground truth info
                arr = data[keys[idx]]

                for obj in arr:
                    loc = obj[:4]
                    label = np.argmax(obj[4:])

                    # transform location for ssd-training
                    loc = corner2center(swap_width_height(loc))

                    actual_locs.append(loc)
                    actual_labels.append(label)

                prepare_loss(pred_confs, pred_locs, actual_labels, actual_locs)
            
            batch_loss = sess.run(loss, feed_dict={input: MINIBATCH, pos: positives, neg: negatives, gt_labels: ex_gt_labels, gt_boxes: ex_gt_boxes})
            BATCH_LOSSES.append(batch_loss)
            sess.run(train_step, feed_dict={input: MINIBATCH, pos: positives, neg: negatives, gt_labels: ex_gt_labels, gt_boxes: ex_gt_boxes})
            print('*** BATCH LOSS: '+str(batch_loss)+' ***')
            print('==================== BATCH: '+str(ba+1)+' END ====================')
        EPOCH_LOSSES.append(np.mean(BATCH_LOSSES))
        print('*** RESULT: '+str(EPOCH_LOSSES[-1]))
        print('==================== EPOCH: '+str(ep+1)+' END ====================')
        
    print('\nEND LEARNING')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.array(range(EPOCH)), EPOCH_LOSSES)
    plt.savefig("loss.png")
    plt.show()

    print('==================== '+str(datetime.datetime.now())+' ====================')