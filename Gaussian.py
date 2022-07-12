# -*- coding: utf-8 -*-
"""
Created on Wed May 11 09:47:57 2022

@author: viani
"""


class Gaussian(object):
    def __init__(self, mean=None, std=None, amp=None):
        self.mean = mean
        self.std = std
        self.amp = amp
