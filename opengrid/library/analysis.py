# -*- coding: utf-8 -*-
"""
General analysis functions.

Try to write all methods such that they take a dataframe as input
and return a dataframe or list of dataframes.
"""

from numbers import Number

import datetime as dt
import pandas as pd
import numpy as np

from opengrid.library.exceptions import EmptyDataFrame


class Analysis():
    """
    Generic Analysis

    An analysis should have a dataframe as input
    self.result should be used as 'output dataframe'
    It also has output methods: plot, to json...
    """

    def __init__(self, data_frame):
        """ Virtual method """
        self.data_frame = data_frame
        self.result = None

    def plot(self, **kwargs):
        """ Virtual method """

    def to_json(self, **kwargs):
        """ Virtual method """


class DailyAgg(Analysis):
    """
    Obtain a dataframe with daily aggregated data according to an aggregation operator
    like min, max or mean
    - for the entire day if starttime and endtime are not specified
    - within a time-range specified by starttime and endtime.
      This can be used eg. to get the minimum consumption during the night.
    """

    def __init__(self: Analysis,
                 data_frame: pd.DataFrame,
                 aggregate: str,
                 starttime: dt.time = dt.time.min,
                 endtime: dt.time = dt.time.max) -> None:
        """
        Parameters
        ----------
        data_frame : pandas.DataFrame
            With pandas.DatetimeIndex and one or more columns
        aggregate : str
            'min', 'max', or another aggregation function
        starttime, endtime : datetime.time objects
            For each day, only consider the time between starttime and endtime
            If None, use begin of day/end of day respectively
        """
        super().__init__(data_frame=data_frame)
        self.aggregate = aggregate
        self.starttime = starttime
        self.endtime = endtime

    def do_analysis(self,
                    aggregate,
                    starttime=dt.time.min,
                    endtime=dt.time.max):
        """ TODO docstring """
        if self.data_frame.empty:
            raise EmptyDataFrame

        data_frame = self.data_frame[(
            self.data_frame.index.time >= starttime
        ) & (
            self.data_frame.index.time < endtime
        )]
        data_frame = data_frame.resample(rule='D',
                                         how=aggregate)
        self.result = data_frame


def standby(data_frame,
            resolution='24h',
            time_window=None):
    """
    Compute standby power

    Parameters
    ----------
    data_frame : pandas.DataFrame or pandas.Series
        Electricity Power
    resolution : str, default='d'
        Resolution of the computation.  Data will be resampled to this resolution (as mean)
        before computation of the minimum.
        String that can be parsed by the pandas resample function, example ='h', '15min', '6h'
    time_window : tuple with start-hour and end-hour, default=None
        Specify the start-time and end-time for the analysis.
        Only data within this time window will be considered.
        Both times have to be specified as string ('01:00', '06:30') or as datetime.time() objects

    Returns
    -------
    data_frame : pandas.Series with DateTimeIndex in the given resolution
    """

    if data_frame.empty:
        raise EmptyDataFrame

    # if data_frame was a pd.Series, convert to DataFrame
    data_frame = pd.DataFrame(data_frame)

    def parse_time(time):
        if isinstance(time, Number):
            return pd.Timestamp.utcfromtimestamp(time).time()
        return pd.Timestamp(time).time()

    # first filter based on the time-window
    if time_window:
        t_start = parse_time(time_window[0])
        t_end = parse_time(time_window[1])
        if t_start > t_end:
            # start before midnight
            data_frame = data_frame[(
                data_frame.index.time >= t_start
            ) | (
                data_frame.index.time < t_end
            )]
        else:
            data_frame = data_frame[(
                data_frame.index.time >= t_start
            ) & (
                data_frame.index.time < t_end
            )]

    return data_frame.resample(resolution).min()


def share_of_standby(data_frame: pd.DataFrame,
                     resolution: str = 'd',
                     time_window=None):
    """
    Compute the share of the standby power in the total consumption.

    Parameters
    ----------
    data_frame : pandas.DataFrame or pandas.Series
        Power (typically electricity, can be anything)
    resolution : str, default='d'
        Resolution of the computation.  Data will be resampled to this resolution (as mean)
        before computation of the minimum.
        String that can be parsed by the pandas resample function, example ='h', '15min', '6h'
    time_window : tuple with start-hour and end-hour, default=None
        Specify the start-time and end-time for the analysis.
        Only data within this time window will be considered.
        Both times have to be specified as string ('01:00', '06:30') or as datetime.time() objects

    Returns
    -------
    fraction : float between 0-1 with the share of the standby consumption
    """

    standby_data_frame = standby(data_frame, resolution, time_window)
    standby_power = standby_data_frame.sum()

    total_data_frame = data_frame.resample(resolution).mean()
    total_power = total_data_frame.sum()

    standby_share = standby_power / total_power
    return standby_share.iloc[0]


def count_peaks(time_series):
    """
    Toggle counter for gas boilers

    Counts the number of times the gas consumption increases with more than 3kW

    Parameters
    ----------
    time_series: Pandas Series
        Gas consumption in minute resolution

    Returns
    -------
    int
    """

    on_toggles = time_series.diff() > 3000
    shifted = np.logical_not(on_toggles.shift(1))
    result = on_toggles & shifted
    count = result.sum()
    return count


def calculate_load_factor(time_series,
                          resolution=None,
                          norm=None):
    """
    Calculate the ratio of input vs. norm over a given interval.

    Parameters
    ----------
    time_series : pandas.Series
        timeseries
    resolution : str, optional
        interval over which to calculate the ratio
        default: resolution of the input timeseries
    norm : int | float, optional
        denominator of the ratio
        default: the maximum of the input timeseries

    Returns
    -------
    pandas.Series
    """

    if not norm:
        norm = time_series.max()

    if resolution:
        time_series = time_series.resample(rule=resolution).mean()

    load_factor = time_series / norm
    return load_factor
