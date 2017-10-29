"""
Bouding Box is the result of comparison with default box.
bouding box has loc and label.

date: 10/4
author: arabian9ts
"""

from model.exception import *

class Box:
    _loc = []
    _index = 0

    def __init__(self, loc, index):
        """
        initializer require all params

        Args:
            locs: where this box exists
            index: class label
        """

        if (loc is None) or (index is None):
            raise NotSpecifiedException('some args', '__init__ @Box')
            
        self._loc = loc
        self._index = index

    @property
    def loc(self):
        return self._loc

    @property
    def index(self):
        return self._index