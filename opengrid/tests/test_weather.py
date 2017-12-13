# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: roel
"""

import unittest
import pandas as pd

import opengrid as og
from opengrid.library import weather
from opengrid.library.exceptions import *


dfw = og.datasets.get('weather_2016_hour')


class WeatherTest(unittest.TestCase):
    def test_compute_degree_days(self):
        res = weather.compute_degree_days(ts=dfw['temperature'],
                                          heating_base_temperatures=[13, 16.5],
                                          cooling_base_temperatures=[16.5, 24])
        self.assertListEqual(sorted(['temp_equivalent', 'HDD_16.5', 'HDD_13', 'CDD_16.5', 'CDD_24']),
                             sorted(res.columns.tolist()))

    def test_compute_degree_days_raises(self):
        df_twodaily = dfw.resample(rule='2D').mean()
        self.assertRaises(UnexpectedSamplingRate, weather.compute_degree_days, df_twodaily, [16], [16])



if __name__ == '__main__':
    unittest.main()