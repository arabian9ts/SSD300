"""
Bouding Box is the result of comparison with default box.
bouding box has loc and label.

date: 10/4
author: arabian9ts
"""

from model.exception import *

class Box:
    _locs = []
    _index = 0

    def __init__(self, loc, index):
        """
        initializer require all params

        Args:
            locs: where this box exists
            index: class label
        """

        if not (loc and index):
            raise NotSpecifiedException('some args', '__init__ @Box')
            
        if isinstance(loc, list) and 4 == len(loc):
            self.loc = loc
        if isinstance(index, int):
            self.index = index

    def loc(self):
        return self._loc

    def index(self):
        return self._index