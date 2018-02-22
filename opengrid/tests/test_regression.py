# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: roel
"""

import unittest
import pandas as pd

import opengrid as og
from opengrid import datasets
import mock

plt_mocked = mock.Mock()
ax_mock = mock.Mock()
fig_mock = mock.Mock()

from opengrid.library.exceptions import EmptyDataFrame


class RegressionTest(unittest.TestCase):

    def test_init(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04)
        mvlr.do_analysis()
        self.assertTrue(hasattr(mvlr, 'list_of_fits'))

    def test_strange_names(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        df_month.rename(columns={'d5a7': '3*tempête !'}, inplace=True)
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04)
        mvlr.do_analysis()
        self.assertTrue(hasattr(mvlr, 'list_of_fits'))

    def test_predict(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        df_month.rename(columns={'d5a7': '3*tempête !'}, inplace=True)
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04)
        mvlr.do_analysis()
        mvlr.add_prediction()

        self.assertListEqual(mvlr.df.columns.tolist(),
                             df_month.columns.tolist() + ['predicted', 'interval_l', 'interval_u'])

    def test_cross_validation(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04, cross_validation=True)
        mvlr.do_analysis()
        self.assertTrue(hasattr(mvlr, 'list_of_fits'))

    def test_prediction(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum().loc['2016', :]
        df_training = df_month.iloc[:-1, :]
        df_pred = df_month.iloc[[-1], :]
        mvlr = og.MultiVarLinReg(df_training, '313b', p_max=0.04)
        mvlr.do_analysis()
        df_pred_95 = mvlr._predict(mvlr.fit, df=df_pred)
        mvlr.confint = 0.98
        df_pred_98 = mvlr._predict(mvlr.fit, df=df_pred)
        self.assertAlmostEqual(df_pred_95.loc['2016-12-01', 'predicted'], df_pred_98.loc['2016-12-01', 'predicted'])
        self.assertTrue(df_pred_98.loc['2016-12-01', 'interval_u'] > df_pred_95.loc['2016-12-01', 'interval_u'])
        self.assertTrue(df_pred_98.loc['2016-12-01', 'interval_l'] < df_pred_95.loc['2016-12-01', 'interval_l'])

        # check limitation to zero
        mvlr.allow_negative_predictions = False
        mvlr.add_prediction()
        self.assertTrue(mvlr.df['predicted'].min() >= 0)

    @mock.patch('opengrid.library.regression.plt', plt_mocked)
    def test_plot(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04)
        mvlr.do_analysis()

        with mock.patch.object(plt_mocked, 'subplots', return_value=(fig_mock, ax_mock)):
            mvlr.plot()

    def test_alternative_metrics(self):
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.04)
        mvlr.do_analysis()
        best_rsquared = mvlr.find_best_rsquared(mvlr.list_of_fits)
        best_akaike = mvlr.find_best_akaike(mvlr.list_of_fits)
        best_bic = mvlr.find_best_bic(mvlr.list_of_fits)
        self.assertEqual(best_rsquared, best_akaike)
        self.assertEqual(best_rsquared, best_bic)

    def test_prune(self):
        "Create overfitted model and prune it"
        df = datasets.get('gas_2016_hour')
        df_month = df.resample('MS').sum()
        mvlr = og.MultiVarLinReg(df_month, '313b')
        mvlr.do_analysis()
        self.assertTrue("ba14" in mvlr.fit.model.exog_names)
        pruned = mvlr._prune(mvlr.fit, 0.05)
        self.assertTrue("ba14" in pruned.model.exog_names)
        pruned = mvlr._prune(mvlr.fit, 0.00009) # with this value, both x will be removed, which is a bit counter-intuitive because initially only ba14 has a pvalue > p_max.
        self.assertFalse("ba14" in pruned.model.exog_names)
        self.assertFalse("d5a7" in pruned.model.exog_names)

        mvlr = og.MultiVarLinReg(df_month, '313b', p_max=0.00009)
        mvlr.do_analysis()
        self.assertFalse("ba14" in mvlr.fit.model.exog_names)
        self.assertFalse("d5a7" in mvlr.fit.model.exog_names)


if __name__ == '__main__':
    unittest.main()
