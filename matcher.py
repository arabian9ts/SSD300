"""
matching function is defined here.

date: 10/2
author: arabian9ts
"""

import numpy as np

from model.policy import *
from model.exception import *
from model.computation import *
from model.bounding_box import Box

class Matcher:
    def __init__(self, fmap_shapes, default_boxes):
        """
        initializer require feature-map shapes and default boxes

        Args:
            fmap_shapes: feature-map's shape
            default_boxes: generated default boxes
        """
        if not fmap_shapes:
            raise NotSpecifiedException('fmap_shapes', 'Matcher __init__')
        if not default_boxes:
            raise NotSpecifiedException('default_boxes', 'Matcher __init__')

        self.fmap_shapes = fmap_shapes
        self.default_boxes = default_boxes
        
    def apply_prediction(self, pred_confs, pred_locs):
        """
        apply prediction to boxes(shaping boxes).

        Args:
            pred_confs: predicated confidences
            pred_locs: predicated locations
        Returns:
            confidences list and anchor boxes
        """
        index = 0
        confs = []
        anchors = [
            [[[None for _ in range(boxes[i])]
            for _ in range(self.fmap_shapes[1])]
            for _ in range(self.fmap_shapes[2])]
            for i in range(len(boxes))
        ]

        for i in range(len(boxes)):
            for y in range(self.fmap_shapes[1]):
                for x in range(self.fmap_shapes[2]):
                    for j in range(boxes[i]):
                        offset = pred_locs[index]
                        cx = self.default_boxes[i][y][x][j][0] + offset[0]
                        cy = self.default_boxes[i][y][x][j][1] + offset[1]
                        height = self.default_boxes[i][y][x][j][2] + offset[2]
                        width = self.default_boxes[i][y][x][j][3] + offset[3]
                        box = [cy, cy, height, width]
                        anchors[i][y][x][j] = box
                        pred_conf = pred_confs[index]
                        prob = np,amax(np.exp(pred_conf) / (np.sum(np.exp(pred_conf)) + 1e-5))
                        pred_label = np.argmax(pred_conf)

                        confs.append(Box(box, prob, pred_label))
                        index += 1
        
        return confs, anchors

    def matching(self, pred_confs, pred_locs, actual_labels, actual_locs):
        """
        match default boxes and bouding boxes.

        Args:
            pred_confs: oredicated confidences
            pred_locs: predicated locations
            actual_labels: answer class labels
            actual_locs: answer box locations
        Returns: 
        """
        
        pos = 0
        matches = [
            [[[None for _ in range(boxes[i])]
            for _ in range(self.fmap_shapes[i][2])]
            for _ in range(self.fmap_shapes[i][1])]
            for i in range(len(boxes))
        ]
        
        # compute jaccard for each default box
        for gt_label, gt_box in zip(actual_labels, actual_locs):
            index = 0
            for i in range(len(boxes)):
                for y in range(self.fmap_shapes[i][1]):
                    for x in range(self.fmap_shapes[i][2]):
                        for j in range(boxes[i]):
                            jacc = jaccard(gt_box, self.default_boxes[index])
                            index += 1

                            if 0.5 <= jacc:
                                # matches[i][y][x][j] = Box(gt_box, gt_label)
                                pos += 1

        neg = 0
        negmax = pos * 3

        return pos
                                