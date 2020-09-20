# -*- coding: utf-8 -*-
"""
General util functions

"""
import datetime
import pandas as pd


def week_schedule(datetime_index,
                  off_days,
                  on_time='9:00',
                  off_time='17:00'):
    """ Return boolean time series following given week schedule.

    Parameters
    ----------
    datetime_index : pandas.DatetimeIndex
        Datetime index
    off_days : list of str
        List of weekdays.
    on_time : str or datetime.time
        Daily opening time. Default: '09:00'
    off_time : str or datetime.time
        Daily closing time. Default: '17:00'

    Returns
    -------
    pandas.Series of bool
        True when on, False otherwise for given datetime index

    Examples
    --------
    >>> import pandas as pd
    >>> from opengrid.library.utils import week_schedule
    >>> datetime_index = pd.date_range('20170701', '20170710', freq='H')
    >>> week_schedule(datetime_index)
    """

    if not isinstance(on_time, datetime.time):
        on_time = pd.to_datetime(on_time, format='%H:%M').time()

    if not isinstance(off_time, datetime.time):
        off_time = pd.to_datetime(off_time, format='%H:%M').time()

    times = (
        datetime_index.time >= on_time
    ) & (
        datetime_index.time < off_time
    ) & (
        ~datetime_index.day_name().isin(off_days)
    )
    return pd.Series(times, index=datetime_index)
