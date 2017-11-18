"""
define in here e.g. the number of class, box scale, ...
meaning constants class

date: 9/30
author: arabian9ts
"""

# the number of classified class
classes = 21

# the number of boxes per feature map
boxes = [6, 6, 6, 6, 6, 6,]

# default box ratios
# each length should be matches boxes[index]
box_ratios = [
    [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0],
    [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0],
    [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0],
    [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0],
    [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0],
    [1.0, 1.0, 2.0, 3.0, 0.5, 1.0/3.0],
]