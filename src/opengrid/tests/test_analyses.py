# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: roel
"""

import unittest
import pandas as pd
import numpy as np

import opengrid as og
from opengrid import datasets
from opengrid.library.exceptions import EmptyDataFrame


class AnalysisTest(unittest.TestCase):

    def test_standby(self):
        df = datasets.get('elec_power_min_1sensor')
        res = og.analysis.standby(df, 'D')
        self.assertEqual(res.index.tz.zone, 'Europe/Brussels')

        self.assertRaises(EmptyDataFrame, og.analysis.standby, pd.DataFrame)

    def test_standby_with_time_window(self):
        df = datasets.get('elec_power_min_1sensor')
        res = og.analysis.standby(df, 'D', time_window=('01:00', '06:00'))
        self.assertEqual(res.index.tz.zone, 'Europe/Brussels')
        self.assertEqual(res.squeeze().to_json(), '{"1507327200000":61.739999936,"1507413600000":214.9799999222,"1507500000000":53.0399997951,"1507586400000":55.7399999164,"1507672800000":59.94000006,"1507759200000":69.4800002407,"1507845600000":56.8200000236,"1507932000000":54.1799997864,"1508018400000":54.779999801,"1508104800000":54.7199997772,"1508191200000":98.5199999576,"1508277600000":55.6799999066,"1508364000000":53.9399997052,"1508450400000":109.5599999931,"1508536800000":144.3600001093,"1508623200000":52.7999997279}')

        res = og.analysis.standby(df, 'D', time_window=('22:00', '06:00'))
        self.assertEqual(res.index.tz.zone, 'Europe/Brussels')
        self.assertEqual(res.squeeze().to_json(), '{"1507327200000":61.739999936,"1507413600000":119.2800000636,"1507500000000":53.0399997951,"1507586400000":55.7399999164,"1507672800000":59.94000006,"1507759200000":69.4800002407,"1507845600000":56.8200000236,"1507932000000":54.1799997864,"1508018400000":54.779999801,"1508104800000":54.7199997772,"1508191200000":98.5199999576,"1508277600000":55.6799999066,"1508364000000":53.9399997052,"1508450400000":96.3000000408,"1508536800000":133.9200000744,"1508623200000":52.7999997279}')

    def test_share_of_standby_1(self):
        df = pd.DataFrame(data={'conso':np.ones(48)},
                          index=pd.DatetimeIndex(start=pd.Timestamp('20180304'), periods=48, freq='h'))
        share_of_standby = og.analysis.share_of_standby(df, resolution='24h')
        self.assertEqual(share_of_standby, 1.0)

    def test_share_of_standby_2(self):
        df = pd.DataFrame(data={'conso':np.ones(48)},
                          index=pd.DatetimeIndex(start=pd.Timestamp('20180304'), periods=48, freq='h'))
        df.iloc[0,0] = 0
        share_of_standby = og.analysis.share_of_standby(df, resolution='24h')
        self.assertAlmostEqual(share_of_standby, 0.5106382978723404)


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
