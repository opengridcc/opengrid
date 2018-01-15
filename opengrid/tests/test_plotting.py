# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: Jan
"""

import unittest


class AnalysisTest(unittest.TestCase):
    def test_plotting(self):
        from opengrid.library.plotting import plot_style
        plt = plot_style()


if __name__ == '__main__':
    unittest.main()
