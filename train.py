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
import tensorflow as tf
import numpy as np
import pickle
import threading
import matplotlib.pyplot as plt

from util.util import *
from tqdm import trange
from model.SSD300 import *

# ====================== Training Parameters ====================== #
BATCH_SIZE = 10
EPOCH = 200
EPOCH_LOSSES = []
SHUFFLED_INDECES = []
# ============================== END ============================== #

if __name__ == '__main__':
    sess = tf.Session()
    buff = []

    # load pickle data set annotation
    with open('VOC2007.pkl', 'rb') as f:
        data = pickle.load(f)
        keys = sorted(data.keys())
        BATCH = int(len(keys) / BATCH_SIZE)

    def next_batch():
        global buff, BATCH_SIZE ,SHUFFLED_INDECES
        mini_batch = []
        actual_data = []

        if 0 == len(SHUFFLED_INDECES):
            SHUFFLED_INDECES = list(np.random.permutation(len(keys)))

        indices = SHUFFLED_INDECES[:min(BATCH_SIZE, len(SHUFFLED_INDECES))]
        del SHUFFLED_INDECES[:min(BATCH_SIZE, len(SHUFFLED_INDECES))]

        for idx in indices:
            # make images mini batch

            img, _, _, _, = preprocess('voc2007/'+keys[idx])

            actual_data.append(data[keys[idx]])
            mini_batch.append(img)

        buff.append((mini_batch, actual_data))


    # tensorflow session
    ssd = SSD300(sess)
    sess.run(tf.global_variables_initializer())

    # parameter saver
    saver = tf.train.Saver()

    # saver.restore(sess, './checkpoints/params.ckpt')

    SHUFFLED_INDECES = list(np.random.permutation(len(keys)))

    print('\nSTART LEARNING')
    print('==================== '+str(datetime.datetime.now())+' ====================')

    for _ in range(5):
        next_batch()
        
    for ep in range(EPOCH):
        BATCH_LOSSES = []
        for ba in trange(BATCH):
            batch, actual = buff.pop(0)
            threading.Thread(name='load', target=next_batch).start()
            _, _, batch_loc, batch_conf, batch_loss = ssd.train(batch, actual)
            BATCH_LOSSES.append(batch_loss)

            # print('BATCH: {0} / EPOCH: {1}, LOSS: {2}'.format(ba+1, ep+1, batch_loss))
        EPOCH_LOSSES.append(np.mean(BATCH_LOSSES))
        print('\n*** AVERAGE: '+str(EPOCH_LOSSES[-1])+' ***')
        saver.save(sess, './checkpoints/params.ckpt')
        print('\n========== EPOCH: '+str(ep+1)+' END ==========')
        
    print('\nEND LEARNING')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(np.array(range(EPOCH)), EPOCH_LOSSES)
    plt.grid()
    plt.savefig("loss.png")
    plt.show()

    print('==================== '+str(datetime.datetime.now())+' ====================')