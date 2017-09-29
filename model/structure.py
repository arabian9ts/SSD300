"""
network-structure of vgg16 excludes fully-connection layer

date: 9/17
author: arabian9ts
"""


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

}

# default kernel_size
kernel_size = [1, 2, 2, 1,]

# default convolution-layer-strides
conv_strides = [1, 1, 1, 1,]

# default pooling-layer-strides
pool_strides = [1, 2, 2, 1,]