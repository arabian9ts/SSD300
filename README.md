# SSD300
Single Shot MultiBox Detector implemented with TensorFlow
## Dependencies ##
python3.6.1
* numpy
* pickle
* skimage
* TensorFlow
* matplotlib

## Usage ##
1. Import required modules
```
import tensorboard as tf
import numpy as np

from model.ssd import *
from util.util import *
from matcher import Matcher
from model.computation import *
from model.default_box import *
```

2. Load test-image  
```
img = load_image('./test.jpg')
img = img.reshape((300, 300, 3))
```

3. Prepare TensorFlow placeholder and Tensor
```
input = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
ssd = SSD()
fmaps, confs, locs = ssd.build(input, is_training=True)
```

4. Prepare default boxes, loss function and matcher  
```
fmap_shapes = [map.get_shape().as_list() for map in fmaps]
dboxes = generate_boxes(fmap_shapes)

# required placeholder for loss
loss, pos, neg, gt_labels, gt_boxes = ssd.loss(len(dboxes))
optimizer = tf.train.AdamOptimizer(0.05)
train_step = optimizer.minimize(loss)

# provides matching method
matcher = Matcher(fmap_shapes, dboxes)
```

5. Running feature-map and predict confs and locs
```
feature_maps, pred_confs, pred_locs = sess.run(train_set, feed_dict={input: img})
```

6. Start Session  
Each default box has self confidence and loc, so expand _len(signal_labels)_ -> _len(default_boxes)_.
_expanded_gt_labels_ and _expanded_gt_boxes_ are them.
```
with tf.Session() as sess:
    batch_loss = sess.run(loss, feed_dict={input: img, pos: positives, neg: negatives, gt_labels: expanded_gt_labels, gt_boxes: expanded_gt_boxes})
    sess.run(train_step, feed_dict={input: img, pos: positives, neg: negatives, gt_labels: expanded_gt_labels, gt_boxes: expanded_gt_boxes})
```

## Test Training ##
```
$ ./setup.sh
$ python train.py
```

## Present Circumstances ##
I just implemented ssd300, so its not already done learning-test.  

If I have overlooked something, please tell me.

## Welcome PullRequest or E-mail ##
