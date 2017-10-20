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

    # tensorflow session
    with tf.Session() as sess:
        ssd = SSD300(sess)
        sess.run(tf.global_variables_initializer())

        # parameter saver
        saver = tf.train.Saver()
        if len(sys.argv) == 2 and 'eval' == sys.argv[2]:
            saver.restore(sess, './checkpoints/params.ckpt')
            minibatch, actual_data = next_batch(is_training=True)
            _, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, False)
            print(batch_loc)
            print(batch_conf)
            print(batch_loss)
            sys.exit()

        print('\nSTART LEARNING')
        print('==================== '+str(datetime.datetime.now())+' ====================')

        for ep in range(EPOCH):
            BATCH_LOSSES = []
            for ba in range(BATCH):
                minibatch, actual_data = next_batch(is_training=True)
                _, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, True)
                BATCH_LOSSES.append(batch_loss)

                print('\n********** BATCH LOSS **********')
                print('\nLOC LOSS:\n'+str(batch_loc))
                print('\nCONF LOSS:\n'+str(batch_conf))
                print('\nTOTAL LOSS: '+str(batch_loss))
                print('\n========== BATCH: '+str(ba+1)+' END ==========')
            EPOCH_LOSSES.append(np.mean(BATCH_LOSSES))
            print('\n*** AVERAGE: '+str(EPOCH_LOSSES[-1])+' ***')

            saver.save(sess, './checkpoints/params.ckpt')

            print('\n*** TEST ***')
            minibatch, actual_data = next_batch(is_training=False)
            _, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, False)
            print('\nLOC LOSS:\n'+str(batch_loc))
            print('\nCONF LOSS:\n'+str(batch_conf))
            print('\nTOTAL LOSS: '+str(batch_loss))
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