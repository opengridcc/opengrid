# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: roel
"""

import unittest
import pandas as pd

import opengrid as og
from opengrid import datasets
from opengrid.library.exceptions import EmptyDataFrame


class AnalysisTest(unittest.TestCase):

    def test_standby(self):
        df = datasets.get('elec_power_min_1sensor')
        res = og.analysis.standby(df, 'D')
        self.assertEqual(res.index.tz.zone, 'Europe/Brussels')

        self.assertRaises(EmptyDataFrame, og.analysis.standby, pd.DataFrame)

    def test_count_peaks(self):
        df = datasets.get('gas_dec2016_min')
        ts = df['313b'].head(100)
        count = og.analysis.count_peaks(ts)
        self.assertEqual(count, 13)


if __name__ == '__main__':
    unittest.main()
