"""
network-structure of vgg16 excludes fully-connection layer

date: 9/17
author: arabian9ts
"""

from policy import *


# structure of convolution and pooling layer up to fully-connection layer
"""
size-format:
             [ [      convolution_kernel     ], [   bias   ] ]
             [ [ f_h, f_w, in_size, out_size ], [ out_size ] ]
"""
structure = {

    # convolution layer 1
    'conv1_1': [[3, 3, 3, 64], [64]],
    'conv1_2': [[3, 3, 64, 64], [64]],

    # convolution layer 2
    'conv2_1': [[3, 3, 64, 128], [128]],
    'conv2_2': [[3, 3, 128, 128], [128]],

    # convolution layer 3
    'conv3_1': [[3, 3, 128, 256], [256]],
    'conv3_2': [[3, 3, 256, 256], [256]],
    'conv3_3': [[3, 3, 256, 256], [256]],

    # convolution layer 4
    'conv4_1': [[3, 3, 256, 512], [512]],
    'conv4_2': [[3, 3, 512, 512], [512]],
    'conv4_3': [[3, 3, 512, 512], [512]],

    # convolution layer 5
    'conv5_1': [[3, 3, 512, 512], [512]],
    'conv5_2': [[3, 3, 512, 512], [512]],
    'conv5_3': [[3, 3, 512, 512], [512]],

    # convolution layer 6
    'conv6': [[3, 3, 512, 1024], [1024]],

    # convolution layer 7
    'conv7': [[1, 1, 1024, 1024], [1024]],

    # convolution layer 8
    'conv8_1': [[1, 1, 1024, 256], [256]],
    'conv8_2': [[3, 3, 256, 512], [512]],

    # convolution layer 9
    'conv9_1': [[1, 1, 512, 128], [128]],
    'conv9_2': [[3, 3, 128, 256], [256]],

    # convolution layer 10
    'conv10_1': [[1, 1, 256, 128], [128]],
    'conv10_2': [[3, 3, 128, 256], [256]],

    # convolution layer 11
    'conv11_1': [[1, 1, 256, 128], [128]],
    'conv11_2': [[3, 3, 128, 256], [256]],

    # feature map 1
    'map1': [[3, 3, 512, boxes[0]*(classes+4)], [boxes[0]*(classes+4)]],

    # feature map 2
    'map2': [[3, 3, 1024, boxes[1]*(classes+4)], [boxes[1]*(classes+4)]],

    # feature map 3
    'map3': [[3, 3, 512, boxes[2]*(classes+4)], [boxes[2]*(classes+4)]],

    # feature map 4
    'map4': [[3, 3, 256, boxes[3]*(classes+4)], [boxes[3]*(classes+4)]],

    # feature map 5
    'map5': [[3, 3, 256, boxes[4]*(classes+4)], [boxes[4]*(classes+4)]],

    # feature map 6
    'map6': [[1, 1, 256, boxes[5]*(classes+4)], [boxes[5]*(classes+4)]],

}

# default kernel_size
kernel_size = [1, 2, 2, 1,]

# default convolution-layer-strides
conv_strides = [1, 1, 1, 1,]

# default pooling-layer-strides
pool_strides = [1, 2, 2, 1,]