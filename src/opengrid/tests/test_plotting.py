# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 02:37:25 2013

@author: Jan
"""

import unittest
import pandas as pd
from opengrid.library import plotting


class PlotStyleTest(unittest.TestCase):
    def test_default(self):
        plt = plotting.plot_style()


class CarpetTest(unittest.TestCase):
    def test_default(self):
        import numpy as np
        index = pd.date_range('2015-1-1', '2015-2-1', freq='h')
        ser = pd.Series(np.random.normal(size=len(index)), index=index, name='abc')
        assert plotting.carpet(ser) is not None

    def test_empty(self):
        assert plotting.carpet(pd.Series(index=list('abc'))) is None

class BoxplotTest(unittest.TestCase):
    def test_default(self):
        import numpy as np
        import pandas as pd
        from opengrid.library import plotting
        index = pd.date_range('2015-1-1', '2015-2-1', freq='d')
        df = pd.DataFrame(index=index, data=np.random.randint(5, size=(len(index),20)))
        plotting.boxplot(df)

    def test_arguments(self):
        import numpy as np
        import pandas as pd
        from opengrid.library import plotting
        index = pd.date_range('2015-1-1', '2015-2-1', freq='d')
        df = pd.DataFrame(index=index, data=np.random.randint(5, size=(len(index),20)))
        plotting.boxplot(df, plot_mean=True, plot_ids=[2, 3], title="Title", xlabel="xlable", ylabel="ylable")


if __name__ == '__main__':
    unittest.main()
