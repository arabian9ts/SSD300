"""
matching function is defined here.
Matcher does matching with only one image.
If use batch, call matching per image.

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
                        box = [cy, cy, width, height]
                        anchors[i][y][x][j] = box
                        pred_conf = pred_confs[index]
                        prob = np,amax(np.exp(pred_conf) / (np.sum(np.exp(pred_conf)) + 1e-5))
                        pred_label = np.argmax(pred_conf)

                        confs.append(Box(box, prob, pred_label))
                        index += 1
        
        return confs, anchors
        

    def extract_highest_indicies(self, pred_confs, max_length):
        """
        extract specific indicies, that is, have most high loss_confs.

        Args:
            pred_confs: predicated confidences
            max_length: max length of extracted indicies (in here, pos*3)
        Returns:
            extracted indicies of boxes (confidences).
        """

        loss_confs = []

        for pred_conf in pred_confs:
            pred = np.exp(pred_conf) / (np.sum(np.exp(pred_conf)) + 1e-5)
            loss_confs.append(np.amax(pred))

        size = min(len(loss_confs), max_length)
        indicies = np.argpartition(loss_confs, -size)[-size:]

        return indicies


    def matching(self, pred_confs, pred_locs, actual_labels, actual_locs):
        """
        match default boxes and bouding boxes.
        matching computes pos and neg count for the computation of loss.
        now, the most noting point is that it is not important that 
        whether class label is correctly predicted.
        class label loss is evaled by loss_conf

        matches variable have some Box instance and most of None.
        if jaccard >= 0.5, that matches box has Box(gt_loc, gt_label).
        then, sort by pred_confs loss and extract 3*pos boxes, which they
        have Box([], classes) => background.

        when compute losses, we need transformed ground truth labels and locations
        because each box has self confidence and location.
        so, we should prepare expanded labels and locations whose size is as same as len(matches).

        Args:
            pred_confs: predicated confidences
            pred_locs: predicated locations
            actual_labels: answer class labels
            actual_locs: answer box locations
        Returns:
            postive_list: if pos -> 1 else 0
            negative_list: if neg and label is not classes(not unknown class) 1 else 0
            expanded_gt_labels: gt_label if pos else classes
            expanded_gt_locs: gt_locs if pos else [0, 0, 0, 0]
        """
        
        pos = 0
        neg = 0
        pos_list = []
        neg_list = []
        expanded_gt_labels = []
        expanded_gt_locs = []
        matches = []

        # generate serializationd matching boxes
        for i in range(len(boxes)):
            for _ in range(self.fmap_shapes[i][1]):
                for _ in range(self.fmap_shapes[i][2]):
                    for _ in range(boxes[i]):
                        matches.append(None)
        
        # compute jaccard for each default box
        for gt_label, gt_box in zip(actual_labels, actual_locs):
            for i in range(len(matches)):
                jacc = jaccard(gt_box, self.default_boxes[i])
                if 0.5 <= jacc:
                    matches[i] = Box(gt_box, gt_label)
                    pos += 1

        indicies = self.extract_highest_indicies(pred_confs, pos*3)

        for i in indicies:
            if classes != np.argmax(pred_confs[i]):
                matches[i] = Box([], classes-1)
                neg += 1

        for i, box in zip(range(len(matches)), matches):
            # if box is None
            # => Neither positive nor negative
            if not box:
                pos_list.append(0)
                neg_list.append(0)
                expanded_gt_labels.append(classes-1)
                expanded_gt_locs.append([0]*4)
            # if box's loc is empty
            # => Negative
            elif not box.loc:
                pos_list.append(0)
                neg_list.append(1)
                expanded_gt_labels.append(classes-1)
                expanded_gt_locs.append([0]*4)
            # if box's loc is specified
            # => Positive
            else:
                pos_list.append(1)
                neg_list.append(0)
                expanded_gt_labels.append(gt_label)
                expanded_gt_locs.append(gt_box)


        return pos_list, neg_list, expanded_gt_labels, expanded_gt_locs