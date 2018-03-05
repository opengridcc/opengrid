# -*- coding: utf-8 -*-
"""
General util functions

"""
import pandas as pd


def week_schedule(index, on_time='9:00', off_time='17:00', off_days=['Sunday', 'Monday']):
    """ Return boolean time series following given week schedule.

    Parameters
    ----------
    index : pandas.DatetimeIndex
        Datetime index
    on_time : str
        Daily opening time. Default: '09:00'
    off_time : str
        Daily closing time. Default: '17:00'
    off_days : list of str
        List of weekdays. Default: ['Sunday', 'Monday']

    Returns
    -------
    pandas.Series of bool
        True when on, False otherwise for given datetime index

    Examples
    --------
    >>> import pandas as pd
    >>> from opengrid.library.utils import week_schedule
    >>> index = pd.date_range('20170701', '20170710', freq='H')
    >>> week_schedule(index)
    """
    on_time = pd.to_datetime(on_time, format='%H:%M').time()
    off_time = pd.to_datetime(off_time, format='%H:%M').time()
    times = (index.time >= on_time) & (index.time < off_time) & (~index.weekday_name.isin(off_days))
    return pd.Series(times, index=index)
