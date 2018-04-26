# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: Jan
"""

import unittest
import pandas as pd
import opengrid as og
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


class LoadDurationCurveTest(unittest.TestCase):
    def test_default(self):
        df = og.datasets.get('gas_2016_hour')
        assert og.plotting.load_duration_curve(df) is not None
        assert og.plotting.load_duration_curve(df, trim_zeros=True) is not None


if __name__ == '__main__':
    unittest.main()
