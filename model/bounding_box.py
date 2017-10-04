"""
Bouding Box is the result of comparison with default box.
bouding box has locs, argmax of softmax output and prediction.

date: 10/4
author: arabian9ts
"""

from model.exception import *

class Box:
    _locs = []
    _prob = 0.0
    _index = 0

    def __init__(self, locs, prob, index):
        """
        initializer require all params

        Args:
            locs: where this box exists
            prob: the most high rate of probability distribution
            index: prob's index (meaning predicted label)
        """

        if not (locs and prob and index):
            raise NotSpecifiedException('some args', '__init__ @Box')
            
        if isinstance(locs, list) and 4 == len(locs):
            self.locs = locs
        if isinstance(prob, float):
            self.prob = prob
        if isinstance(index, int):
            self.index = index

    def locs(self):
        return self._locs

    def prob(self):
        return self._prob

    def index(self):
        return self._index