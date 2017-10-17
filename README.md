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

from util.util import *
from model.SSD300 import *
```

2. Load test-image  
```
img = load_image('./test.jpg')
img = img.reshape((300, 300, 3))
```

3. Prepare TensorFlow placeholder (input images)
```
input = tf.placeholder(shape=[None, 300, 300, 3], dtype=tf.float32)
```

4. Start Session  
you must just call ssd.eval() !
```
with tf.Session() as sess:
        ssd = SSD300(sess)
        sess.run(tf.global_variables_initializer())
        for ep in range(EPOCH):
            BATCH_LOSSES = []
            for ba in range(BATCH):
                minibatch, actual_data = next_batch(is_training=True)
                _, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, True)
```

## Test Training ##
you have to extract data-set from zip files.
decompress all zip files in datasets/ and move to VOC2007/ dir.
```
$ ls VOC2007/ | wc -l    #  => 4954
$ ./setup.sh
$ python train.py
```

## Present Circumstances ##
I just implemented ssd300, so its not already done learning-test.  

If I have overlooked something, please tell me.

## Welcome PullRequest or E-mail ##
