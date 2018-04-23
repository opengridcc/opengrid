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

    def test_load_factor(self):
        ts = og.datasets.get('electricity_2016_hour')
        ts = ts['e1de'].truncate(after=pd.Timestamp('20160107'))
        lf1 = og.analysis.load_factor(ts)
        self.assertIsInstance(ts, pd.Series)
        self.assertAlmostEqual(ts.iloc[0], (lf1 * ts.max()).iloc[0])

        lf2 = og.analysis.load_factor(ts, resolution='3h', norm=800)
        self.assertIsInstance(ts, pd.Series)
        self.assertAlmostEqual(175.0345212009457, (lf2 * 800).iloc[0])


if __name__ == '__main__':
    unittest.main()
