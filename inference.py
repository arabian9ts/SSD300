"""
inference script

date: 3/17
author: arabian9ts
"""

import cv2
import sys
from util.util import *
from model.SSD300 import *

def inference(image_name):
    if image_name is None:
        return Exception('not specified image name to be drawed')

    fontType = cv2.FONT_HERSHEY_SIMPLEX
    img, w, h, _, = preprocess('./voc2007/'+image_name)
    pred_confs, pred_locs = ssd.infer(images=[img])
    locs, labels = ssd.ssd.detect_objects(pred_confs, pred_locs)
    img = deprocess(img)
    if len(labels) and len(locs):
        for label, loc in zip(labels, locs):
            loc = center2corner(loc)
            loc = convert2diagonal_points(loc)
            cv2.rectangle(img, (int(loc[0]*w), int(loc[1]*h)), (int(loc[2]*w), int(loc[3]*h)), (0, 0, 255), 1)
            cv2.putText(img, str(int(label)), (int(loc[0]*w), int(loc[1]*h)), fontType, 0.7, (0, 0, 255), 1)

    return img


# detect objects on a specified image.
if 2 == len(sys.argv):
    sess = tf.Session()
    # tensorflow session
    ssd = SSD300(sess)
    sess.run(tf.global_variables_initializer())

    # parameter saver
    saver = tf.train.Saver()
    saver.restore(sess, './checkpoints/params.ckpt')
    img = inference(sys.argv[1])
    cv2.imwrite('./evaluated/'+sys.argv[1], img)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit()