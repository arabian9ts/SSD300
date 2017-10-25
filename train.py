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

import gc
import cv2
import sys
import datetime
import tensorboard as tf
import numpy as np
import pickle
import matplotlib.pyplot as plt

from util.util import *
from model.SSD300 import *

# ====================== Training Parameters ====================== #
BATCH_SIZE = 100
EPOCH = 50
EPOCH_LOSSES = []
# ============================== END ============================== #

if __name__ == '__main__':

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


    def draw_marker(image_name, save):
        if image_name is None:
            return Exception('not specified image name to be drawed')

        img = cv2.imread('./voc2007/'+image_name, 1)
        h = img.shape[0]
        w = img.shape[1]
        reshaped = cv2.resize(img, (300, 300))
        reshaped = reshaped / 255
        pred_confs, pred_locs = ssd.eval(images=[reshaped], actual_data=None, is_training=False)
        labels, locs = ssd.ssd.detect_objects(pred_confs, pred_locs)
        if len(labels) and len(locs):
            for label, loc in zip(labels, locs):
                loc = center2corner(loc)
                loc = convert2diagonal_points(loc)
                cv2.rectangle(img, (int(loc[0]*w), int(loc[1]*h)), (int(loc[2]*w), int(loc[3]*h)), (0, 0, 255), 1)

        if save:
            if not os.path.exists('./evaluated'):
                os.mkdir('./evaluated')
            cv2.imwrite('./evaluated/'+image_name, img)

        return img


    # tensorflow session
    with tf.Session() as sess:
        ssd = SSD300(sess)
        sess.run(tf.global_variables_initializer())

        # parameter saver
        saver = tf.train.Saver()

        # eval and predict object on a specified image.
        if 2 == len(sys.argv):
            saver.restore(sess, './checkpoints/params.ckpt')
            img = draw_marker(sys.argv[1], save=False)
            cv2.namedWindow("img", cv2.WINDOW_NORMAL)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            sys.exit()

        print('\nSTART LEARNING')
        print('==================== '+str(datetime.datetime.now())+' ====================')

        for ep in range(EPOCH):
            BATCH_LOSSES = []
            for ba in range(BATCH):
                minibatch, actual_data = next_batch(is_training=True)
                _, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, True)
                BATCH_LOSSES.append(batch_loss)
                del minibatch
                gc.collect()

                print('\n********** BATCH LOSS **********')
                print('\nLOC LOSS:\n'+str(batch_loc))
                print('\nCONF LOSS:\n'+str(batch_conf))
                print('\nTOTAL LOSS: '+str(batch_loss))
                print('\n========== BATCH: '+str(ba+1)+' END ==========')
            EPOCH_LOSSES.append(np.mean(BATCH_LOSSES))
            print('\n*** AVERAGE: '+str(EPOCH_LOSSES[-1])+' ***')

            saver.save(sess, './checkpoints/params.ckpt')

            print('\n*** TEST ***')
            id = np.random.choice(len(test_keys))
            name = test_keys[id]
            draw_marker(image_name=name, save=True)
            print('\nSaved Evaled Image\n')
            print('\n========== EPOCH: '+str(ep+1)+' END ==========')
            
        print('\nEND LEARNING')

        
        saver.save(sess, './params_final.ckpt')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(np.array(range(EPOCH)), EPOCH_LOSSES)
        plt.grid()
        plt.savefig("loss.png")
        plt.show()

        print('==================== '+str(datetime.datetime.now())+' ====================')