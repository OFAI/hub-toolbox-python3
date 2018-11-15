# -*- coding: utf-8 -*-
"""
This file is part of the HUB TOOLBOX available at
https://github.com/OFAI/hub-toolbox-python3/
The HUB TOOLBOX is licensed under the terms of the GNU GPLv3.

(c) 2018, Roman Feldbauer
Austrian Research Institute for Artificial Intelligence (OFAI)
Contact: <roman.feldbauer@ofai.at>
"""
from multiprocessing import Value

__all__ = ['SynchronizedCounter']

class SynchronizedCounter(object):
    """ A multiprocessing-safe counter for progress information. """
    def __init__(self, init:int=-1):
        self.val = Value('i', init)

    def increment_and_get_value(self, n=1) -> int:
        """ Obtain a lock before incrementing, since += isn't atomic. """
        with self.val.get_lock():
            self.val.value += n
            return self.val.value

    @property
    def value(self) -> int:
        return self.val.value
