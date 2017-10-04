"""
exceptions are defined here.

date: 10/4
author: arabian9ts
"""

class NotSpecifiedException(Exception):
    def __init__(self, name, epicenter):
        self.name = name
        self.epicenter = epicenter
    def __str__(self):
        return ('Parameter "{0.name}" is not specified in "{0.epicenter}"'.format(self))