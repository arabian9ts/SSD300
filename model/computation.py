"""
compute required indicies.

date: 10/1
author: arabian9ts
"""


def intersection(rect1, rect2):
    """
    intersecton of units
    compute boarder line top, left, right and bottom.
    rect is defined as [ top_left_x, top_left_y, width, height ]
    """
    top = max(rect1[1], rect2[1])
    left = max(rect1[0], rect2[0])
    right = min(rect1[0] + rect1[2], rect2[0] + rect2[2])
    bottom = min(rect1[1] + rect1[3], rect2[1] + rect2[3])

    if bottom > top and right > left:
        return (bottom-top)*(right-left)

    return 0


def jaccard(rect1, rect2):
    """
    Jaccard index.
    Jaccard index is defined as #(A∧B) / #(A∨B)
    """
    rect1_ = [x if x >= 0 else 0 for x in rect1]
    rect2_ = [x if x >= 0 else 0 for x in rect2]
    s = rect1_[2]*rect1_[3] + rect2_[2]*rect2_[3]

    # rect1 and rect2 => A∧B
    intersect = intersection(rect1_, rect2_)

    # rect1 or rect2 => A∨B
    union = s - intersect

    # A∧B / A∨B
    return intersect / union


def corner2center(rect):
    """
    rect is defined as [ top_left_x, top_left_y, width, height ]
    """
    center_x = (2 * rect[0] + rect[2]) * 0.5
    center_y = (2 * rect[1] + rect[3]) * 0.5

    return [center_x, center_y, abs(rect[2]), abs(rect[3])]


def center2corner(rect):
    """
    rect is defined as [ top_left_x, top_left_y, width, height ]
    """
    corner_x = rect[0] - rect[2] * 0.5
    corner_y = rect[1] - rect[3] * 0.5

    return [corner_x, corner_y, abs(rect[2]), abs(rect[3])]


def convert2diagonal_points(rect):
    """
    convert rect format

    Args:
        input format is...
        [ top_left_x, top_left_y, width, height ]
    Returns:
        output format is...
        [ top_left_x, top_left_y, bottom_right_x, bottom_right_y ]
    """
    return [rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]]


def convert2wh(rect):
    """
    convert rect format

    Args:
        input format is...
        [ top_left_x, top_left_y, bottom_right_x, bottom_right_y ]
    Returns:
        output format is...
        [ top_left_x, top_left_y, width, height ]
    """
    return [rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]]