"""
Defalut Box generator.

date: 9/30
author: arabian9ts
"""

import numpy as np
from model.policy import *

def scale(k):
    """
    compute default box scale.
    In SSD-Thesis, s_k is defined as
        s_k = s_min + (s_max - s_min) * (k - 1.0) / (m - 1.0), m = 6
    I corresponded with index from zero, so replace (k - 1.0) -> k.

    Args: feature map number
    Returns: scale
    """
    s_min = 0.1
    s_max = 0.9
    m = 6.0
    s_k = s_min + (s_max - s_min) * k / (m - 1.0)
    return s_k


def generate_boxes(fmap_shapes):
    """
    generate default boxes based on defined number
    the shape is [  first-feature-map-boxes ,
                    second-feature-map-boxes ,
                                ...
                    sixth-feature-map-boxes , ]
        ==> ( total_boxes_number x defined_size )

    Args:
        feature map sizes per output such as...
        [ [ 19, 19, ],      # feature-map-shape 1
          [ 19, 19, ],      # feature-map-shape 2
          [ 10, 10 ],       # feature-map-shape 3
          [ 5, 5, ],        # feature-map-shape 4
          [ 3, 3, ],        # feature-map-shape 5
          [ 1, 1, ],        # feature-map-shape 6
        ]
    Returns:
        generated default boxes list
    """

    default_boxes = []

    # this loop should be already 6 loops
    for index, map_shape in enumerate(fmap_shapes):
        s_k = scale(index)
        s_k1 = scale(index+1)
        height = map_shape[1]
        width = map_shape[2]
        ratios = box_ratios[index]
        s = 0.0

        for y in range(height):
            center_y = (y + 0.5) / float(height)
            for x in range(width):
                center_x = (x + 0.5) / float(width)
                for i, ratio in enumerate(ratios):
                    s = s_k

                    if 0 == i:
                        s = np.sqrt(s_k*s_k1)
                    
                    box_width = s * np.sqrt(ratio)
                    box_height = s / np.sqrt(ratio)

                    default_boxes.append([center_x, center_y, box_width, box_height])
                    
    return default_boxes