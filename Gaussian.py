# -*- coding: utf-8 -*-

# Author: Alessandro Viani <viani@dima.unige.it>
#
# License: BSD (3-clause)

class Gaussian(object):
    """Single current dipole class for SESAME.

       Parameters
       ----------
       mean : :py:class:`double`
           The gaussian mean.
       std : :py:class:`double`
           The gaussian standard deviation.
       amp : :py:class:`double`
           The gaussian amplitude or weight.
       """
    def __init__(self, mean=None, std=None, amp=None):
        self.mean = mean
        self.std = std
        self.amp = amp
