"""
matching function is defined here.

date: 10/2
author: arabian9ts
"""

class Matcher:
    def __init__(self, fmap_shapes, default_boxes):
        if not fmap_shapes:
            raise NotSpecifiedException('fmap_shapes', 'Matcher __init__')
        if not default_boxes:
            raise NotSpecifiedException('default_boxes', 'Matcher __init__')
        
    def apply_prediction(pred_confs, pred_locs, fmap_shapes):
        return 

class NotSpecifiedException(Exception):
    def __init__(self, name, epicenter):
        self.name = name
        self.epicenter = epicenter
    def __str__(self):
        return ('Parameter "{0.name}" is not specified in "{0.epicenter}"'.format(self))
