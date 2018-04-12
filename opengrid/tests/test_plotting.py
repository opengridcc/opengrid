# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: Jan
"""

import unittest


class PlotStyleTest(unittest.TestCase):
    def test_default(self):
        from opengrid.library.plotting import plot_style
        plt = plot_style()


class CarpetTest(unittest.TestCase):
    def test_default(self):
        import numpy as np
        import pandas as pd
        from opengrid.library import plotting
        index = pd.date_range('2015-1-1', '2015-12-31', freq='h')
        ser = pd.Series(np.random.normal(size=len(index)), index=index, name='abc')
        plotting.carpet(ser)


if __name__ == '__main__':
    unittest.main()
