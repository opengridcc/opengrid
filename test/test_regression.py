# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: roel
"""

import pickle
import unittest
import mock

from opengrid.library import regression
from datasets import datasets

plt_mocked = mock.Mock()
ax_mock = mock.Mock()
fig_mock = mock.Mock()


class RegressionTest(unittest.TestCase):

    def test_init(self):
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum()
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.04})
        mvlr.do_analysis()
        self.assertTrue(hasattr(mvlr, 'list_of_fits'))

    def test_raises(self):
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum()
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.04})
        self.assertRaises(UnboundLocalError, mvlr.add_prediction)
        try:
            x = mvlr.list_of_fits
            self.assertTrue(False)
        except UnboundLocalError:
            self.assertTrue(True)

    def test_strange_names(self):
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum()
        data_frame_month.rename(columns={'d5a7': '3*tempête !'}, inplace=True)
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.04})
        mvlr.do_analysis()
        self.assertTrue(hasattr(mvlr, 'list_of_fits'))

    def test_predict(self):
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum()
        data_frame_month.rename(columns={'d5a7': '3*tempête !'}, inplace=True)
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.04})
        mvlr.do_analysis()
        mvlr.add_prediction()

        self.assertListEqual(mvlr.data_frame.columns.tolist(),
                             data_frame_month.columns.tolist() + ['predicted', 'interval_l', 'interval_u'])

    def test_cross_validation(self):
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum()
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={
                                             'p_max': 0.04,
                                             'cross_validation': True
                                         })
        mvlr.do_analysis()
        self.assertTrue(hasattr(mvlr, 'list_of_fits'))

    def test_prediction(self):
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum().loc['2016', :]
        data_frame_training = data_frame_month.iloc[:-1, :]
        data_frame_pred = data_frame_month.iloc[[-1], :]
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.04})
        mvlr.do_analysis()
        data_frame_pred_95 = mvlr._predict(mvlr.fit, data_frame=data_frame_pred)
        mvlr.confint = 0.98
        data_frame_pred_98 = mvlr._predict(mvlr.fit, data_frame=data_frame_pred)
        self.assertAlmostEqual( data_frame_pred_95.loc['2016-12-01', 'predicted'], 
                                data_frame_pred_98.loc['2016-12-01', 'predicted'])
        self.assertTrue(data_frame_pred_98.loc['2016-12-01', 'interval_u'] >= 
                        data_frame_pred_95.loc['2016-12-01', 'interval_u'])
        self.assertTrue(data_frame_pred_98.loc['2016-12-01', 'interval_l'] <= 
                        data_frame_pred_95.loc['2016-12-01', 'interval_l'])

        # check limitation to zero
        mvlr.allow_negative_predictions = False
        mvlr.add_prediction()
        self.assertTrue(mvlr.data_frame['predicted'].min() >= 0)

    @mock.patch('opengrid.library.regression.pyplot', plt_mocked)
    def test_plot(self):
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum()
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.04})
        mvlr.do_analysis()

        with mock.patch.object(plt_mocked, 'subplots', return_value=(fig_mock, ax_mock)):
            mvlr.plot()

    def test_alternative_metrics(self):
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum()
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.04})
        mvlr.do_analysis()
        best_rsquared = mvlr.find_best_rsquared(mvlr.list_of_fits)
        best_akaike = mvlr.find_best_akaike(mvlr.list_of_fits)
        best_bic = mvlr.find_best_bic(mvlr.list_of_fits)
        self.assertEqual(best_rsquared, best_akaike)
        self.assertEqual(best_rsquared, best_bic)

    def test_prune(self):
        "Create overfitted model and prune it"
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum()
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b')
        mvlr.do_analysis()
        self.assertTrue("ba14" in mvlr.fit.model.exog_names)
        pruned = mvlr._prune(mvlr.fit, 0.05)
        self.assertTrue("ba14" in pruned.model.exog_names)
        # with this value, both x will be removed, which is a bit counter-intuitive because initially only ba14 has a pvalue > p_max.
        pruned = mvlr._prune(mvlr.fit, 0.00009)
        self.assertFalse("ba14" in pruned.model.exog_names)
        self.assertFalse("d5a7" in pruned.model.exog_names)

        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.00009})
        mvlr.do_analysis()
        self.assertFalse("ba14" in mvlr.fit.model.exog_names)
        self.assertFalse("d5a7" in mvlr.fit.model.exog_names)

    def test_pickle_round_trip(self):
        "Pickle, unpickle and check results"
        data_frame = datasets.get('gas_2016_hour')
        data_frame_month = data_frame.resample('MS').sum().loc['2016', :]
        data_frame_training = data_frame_month.iloc[:-1, :]
        data_frame_pred = data_frame_month.iloc[[-1], :]
        mvlr = regression.MultiVarLinReg(data_frame=data_frame_month,
                                         dependent_var='313b',
                                         options={'p_max': 0.04})
        mvlr.do_analysis()
        data_frame_pred_95_orig = mvlr._predict(mvlr.fit, data_frame=data_frame_pred)

        s = pickle.dumps(mvlr)
        m = pickle.loads(s)
        self.assertTrue(hasattr(m, 'list_of_fits'))
        data_frame_pred_95_roundtrip = m._predict(
            m.fit, data_frame=data_frame_pred)
        self.assertAlmostEqual( data_frame_pred_95_orig.loc['2016-12-01', 'predicted'], 
                                data_frame_pred_95_roundtrip.loc['2016-12-01', 'predicted'])


if __name__ == '__main__':
    unittest.main()
