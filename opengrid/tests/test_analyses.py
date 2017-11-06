# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 09:36:16 2016

@author: roel
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: roel
"""

import unittest
import pandas as pd

from opengrid.library import analysis
from opengrid import datasets
from opengrid.library.exceptions import EmptyDataFrameError

class AnalysisTest(unittest.TestCase):
    
    def test_standby(self):
        df = datasets.get('elec_power_min_1sensor')
        res = analysis.standby(df, 'D')
        self.assertEqual(res.index.tz.zone, 'Europe/Brussels')

        self.assertRaises(EmptyDataFrameError, analysis.standby, pd.DataFrame)



if __name__ == '__main__':
    
    unittest.main()