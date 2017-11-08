# SSD300
Single Shot MultiBox Detector implemented with TensorFlow
## Dependencies ##
python3.6.1
* numpy
* skimage
* TensorFlow
* matplotlib
* OpenCV

## Usage ##
1. Import required modules
```
import tensorflow as tf
import numpy as np

from util.util import *
from model.SSD300 import *
```

2. Load test-image  
```
img = load_image('./test.jpg')
img = img.reshape((300, 300, 3))
```

3. Start Session  
```
with tf.Session() as sess:
        ssd = SSD300(sess)
        sess.run(tf.global_variables_initializer())
        for ep in range(EPOCH):
            ...
```

4. Training or Evaluating
you must just call ssd.eval() !
```
...

_, _, batch_loc, batch_conf, batch_loss = ssd.eval(minibatch, actual_data, is_training=True)

...
```


## Test Training ##
you have to extract data-set from zip files.
decompress all zip files in datasets/ and move to voc2007/ dir.
```
$ ls voc2007/ | wc -l    #  => 4954
$ ./setup.sh
$ python train.py
```

## Present Circumstances ##
I'm checking and testing SSD model, so this model may not be complete.

If I have overlooked something, please tell me.

## Welcome PullRequest or E-mail ##
