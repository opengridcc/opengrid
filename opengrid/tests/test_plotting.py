# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: Jan
"""

import unittest
import pandas as pd
from opengrid.library import plotting


class PlotStyleTest(unittest.TestCase):
    def test_default(self):
        plt = plotting.plot_style()


class CarpetTest(unittest.TestCase):
    def test_default(self):
        import numpy as np
        index = pd.date_range('2015-1-1', '2015-12-31', freq='h')
        ser = pd.Series(np.random.normal(size=len(index)), index=index, name='abc')
        assert plotting.carpet(ser) is not None

    def test_empty(self):
        assert plotting.carpet(pd.Series(index=list('abc'))) is None


if __name__ == '__main__':
    unittest.main()
