# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: roel
"""

import unittest
import pandas as pd

import opengrid as og
from opengrid import datasets
from opengrid.library.exceptions import EmptyDataFrameError


class RegressionTest(unittest.TestCase):
    
    def test_init(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04)
        self.assertTrue(hasattr(mvlr, 'list_of_fits'))

    def test_strange_names(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        df_month.rename(columns={'d5a7':'3*tempête !'}, inplace=True)
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04)
        self.assertTrue(hasattr(mvlr, 'list_of_fits'))

    def test_predict(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        df_month.rename(columns={'d5a7':'3*tempête !'}, inplace=True)
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04)
        mvlr.predict()

        self.assertListEqual(mvlr.df.columns.tolist(),
                             df_month.columns.tolist() + ['Intercept', 'predicted', 'interval_l', 'interval_u'])





if __name__ == '__main__':
    unittest.main()