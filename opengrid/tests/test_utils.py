# -*- coding: utf-8 -*-
"""

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import datetime
import pandas as pd
from opengrid.library import utils


class WeekScheduleTest(unittest.TestCase):

    def test_week_schedule_9to5(self):
        index = pd.date_range('20170701', '20170708', freq='H')
        self.assertEqual(utils.week_schedule(index).sum(), 40)

    def test_week_schedule_7on7(self):
        index = pd.date_range('20170701', '20170708', freq='H')
        self.assertEqual(utils.week_schedule(index, off_days=[]).sum(), 56)

    def test_week_schedule_halftime(self):
        index = pd.date_range('20170701', '20170708', freq='H')
        self.assertEqual(utils.week_schedule(index, off_time='13:00').sum(), 20)

    def test_week_schedule_halftime_with_datetime(self):
        index = pd.date_range('20170701', '20170708', freq='H')
        on_time = datetime.time(hour=9)
        off_time = datetime.time(hour=13)
        self.assertEqual(utils.week_schedule(index, on_time=on_time, off_time=off_time).sum(), 20)


if __name__ == '__main__':
    unittest.main()
