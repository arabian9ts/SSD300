"""
utility function group
such as load_images

date: 9/17
author: arabian9ts
"""

import numpy
import skimage
import skimage.io
import skimage.transform

def load_image(path):
    """
    load specified image

    Args: image path
    Return: resized image
    """
    img = skimage.io.imread(path)
    img = img / 255.
    resized_img = skimage.transform.resize(img, (224, 224))
    return numpy.array(resized_img, dtype=numpy.float32)
