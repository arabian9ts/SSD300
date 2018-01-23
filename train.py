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

import cv2
import sys
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
EPOCH = 100
EPOCH_LOSSES = []
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
        global buff
        mini_batch = []
        actual_data = []
        indicies = np.random.choice(len(keys), BATCH_SIZE)

        for idx in indicies:
            # make images mini batch

            img = load_image('voc2007/'+keys[idx])

            actual_data.append(data[keys[idx]])
            mini_batch.append(img)

        buff.append((mini_batch, actual_data))


    def draw_marker(image_name, save):
        if image_name is None:
            return Exception('not specified image name to be drawed')

        img = cv2.imread('./voc2007/'+image_name, 1)
        h = img.shape[0]
        w = img.shape[1]
        reshaped = cv2.resize(img, (300, 300))
        reshaped = reshaped / 255
        pred_confs, pred_locs = ssd.eval(images=[reshaped], actual_data=None, is_training=False)
        locs, labels = ssd.ssd.detect_objects(pred_confs, pred_locs)
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

    # saver.restore(sess, './checkpoints/params.ckpt')

    print('\nSTART LEARNING')
    print('==================== '+str(datetime.datetime.now())+' ====================')

    for _ in range(5):
        next_batch()

    for ep in range(EPOCH):
        BATCH_LOSSES = []
        for ba in trange(BATCH):
            batch, actual = buff.pop(0)
            threading.Thread(name='load', target=next_batch).start()
            _, _, batch_loc, batch_conf, batch_loss = ssd.eval(batch, actual, True)
            BATCH_LOSSES.append(batch_loss)

            # print('BATCH: {0} / EPOCH: {1}, LOSS: {2}'.format(ba+1, ep+1, batch_loss))
        EPOCH_LOSSES.append(np.mean(BATCH_LOSSES))
        print('\n*** AVERAGE: '+str(EPOCH_LOSSES[-1])+' ***')

        saver.save(sess, './checkpoints/params.ckpt')

        
        print('\n*** TEST ***')
        id = np.random.choice(len(keys))
        name = keys[id]
        draw_marker(image_name=name, save=True)
        print('\nSaved Evaled Image')
        

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