# -*- coding: utf-8 -*-
"""
General analysis functions.

Try to write all methods such that they take a dataframe as input
and return a dataframe or list of dataframes.
"""

import datetime as dt
import pandas as pd
import numpy as np
import numbers
from opengrid.library.exceptions import EmptyDataFrame
from collections import namedtuple


class Analysis(object):
    """
    Generic Analysis

    An analysis should have a dataframe as input
    self.result should be used as 'output dataframe'
    It also has output methods: plot, to json...
    """

    def __init__(self, df, *args, **kwargs):
        self.df = df
        self.do_analysis(*args, **kwargs)

    def do_analysis(self, *args, **kwargs):
        # To be overwritten by inheriting class
        self.result = self.df.copy()

    def plot(self):
        self.result.plot()

    def to_json(self):
        return self.result.to_json()


class DailyAgg(Analysis):
    """
    Obtain a dataframe with daily aggregated data according to an aggregation operator
    like min, max or mean
    - for the entire day if starttime and endtime are not specified
    - within a time-range specified by starttime and endtime.
      This can be used eg. to get the minimum consumption during the night.
    """

    def __init__(self, df, agg, starttime=dt.time.min, endtime=dt.time.max):
        """
        Parameters
        ----------
        df : pandas.DataFrame
            With pandas.DatetimeIndex and one or more columns
        agg : str
            'min', 'max', or another aggregation function
        starttime, endtime : datetime.time objects
            For each day, only consider the time between starttime and endtime
            If None, use begin of day/end of day respectively
        """
        super(DailyAgg, self).__init__(df, agg, starttime=starttime, endtime=endtime)

    def do_analysis(self, agg, starttime=dt.time.min, endtime=dt.time.max):
        if not self.df.empty:
            df = self.df[(self.df.index.time >= starttime) & (self.df.index.time < endtime)]
            df = df.resample('D', how=agg)
            self.result = df
        else:
            self.result = pd.DataFrame()


def standby(df, resolution='24h', time_window=None):
    """
    Compute standby power

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        Electricity Power
    resolution : str, default='d'
        Resolution of the computation.  Data will be resampled to this resolution (as mean) before computation
        of the minimum.
        String that can be parsed by the pandas resample function, example ='h', '15min', '6h'
    time_window : tuple with start-hour and end-hour, default=None
        Specify the start-time and end-time for the analysis.
        Only data within this time window will be considered.
        Both times have to be specified as string ('01:00', '06:30') or as datetime.time() objects

    Returns
    -------
    df : pandas.Series with DateTimeIndex in the given resolution
    """

    if df.empty:
        raise EmptyDataFrame()

    df = pd.DataFrame(df)  # if df was a pd.Series, convert to DataFrame
    def parse_time(t):
        if isinstance(t, numbers.Number):
            return pd.Timestamp.utcfromtimestamp(t).time()
        else:
            return pd.Timestamp(t).time()


    # first filter based on the time-window
    if time_window is not None:
        t_start = parse_time(time_window[0])
        t_end = parse_time(time_window[1])
        if t_start > t_end:
            # start before midnight
            df = df[(df.index.time >= t_start) | (df.index.time < t_end)]
        else:
            df = df[(df.index.time >= t_start) & (df.index.time < t_end)]

    return df.resample(resolution).min()


def share_of_standby(df, resolution='24h', time_window=None):
    """
    Compute the share of the standby power in the total consumption.

    Parameters
    ----------
    df : pandas.DataFrame or pandas.Series
        Power (typically electricity, can be anything)
    resolution : str, default='d'
        Resolution of the computation.  Data will be resampled to this resolution (as mean) before computation
        of the minimum.
        String that can be parsed by the pandas resample function, example ='h', '15min', '6h'
    time_window : tuple with start-hour and end-hour, default=None
        Specify the start-time and end-time for the analysis.
        Only data within this time window will be considered.
        Both times have to be specified as string ('01:00', '06:30') or as datetime.time() objects

    Returns
    -------
    fraction : float between 0-1 with the share of the standby consumption
    """

    p_sb = standby(df, resolution, time_window)
    df = df.resample(resolution).mean()
    p_tot = df.sum()
    p_standby = p_sb.sum()
    share_standby = p_standby / p_tot
    res = share_standby.iloc[0]
    return res


def count_peaks(ts):
    """
    Toggle counter for gas boilers

    Counts the number of times the gas consumption increases with more than 3kW

    Parameters
    ----------
    ts: Pandas Series
        Gas consumption in minute resolution

    Returns
    -------
    int
    """

    on_toggles = ts.diff() > 3000
    shifted = np.logical_not(on_toggles.shift(1))
    result = on_toggles & shifted
    count = result.sum()
    return count


def load_factor(ts, resolution=None, norm=None):
    """
    Calculate the ratio of input vs. norm over a given interval.

    Parameters
    ----------
    ts : pandas.Series
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
    if norm is None:
        norm = ts.max()

    if resolution is not None:
        ts = ts.resample(rule=resolution).mean()

    lf = ts / norm

    return lf


def load_duration(df, trim_zeros=False):
    """
    Create descending load duration series
    (mainly for use in a load duration curve)

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
    trim_zeros : bool
        trim trailing zero's

    Returns
    -------
    pd.DataFrame or pd.Series
    """
    df = pd.DataFrame(df)  # in case a series is passed, wrap it in a dataframe
    load_durations = (df[column].reset_index(drop=True).sort_values(ascending=False).reset_index(drop=True) for column in df)
    if trim_zeros:
        load_durations = (np.trim_zeros(s, trim='b') for s in load_durations)
    df = pd.concat(load_durations, axis=1)
    result = df.squeeze()
    return result


def modulation_detection(ts, min_level=0.1):
    """
    Detect the modulation levels of a gas boiler

    Parameters
    ----------
    ts : pd.Series
    min_level : float
        Physically, a gas boiler cannot modulate under a certain percentage of its maximum power
        So we use this percentage to cut off any noise

    Returns
    -------
    namedtuple(median, minimum)
    """
    # drop all values below the minimum level
    ts = ts[ts >= (ts.max() * min_level)]

    # load duration curve
    ld = load_duration(ts, trim_zeros=True)

    # find the part in the load duration curve with the highest number of consecutive identical values
    # a.k.a. find the longest 'flat part' in the curve
    median_modul = ld.round().groupby(ld.round()).size().sort_values(ascending=False).index[0]

    # take the second derivative of the whatever happens after the flat part
    dif2 = ld[ld < median_modul].diff().diff().shift(-2)
    # find the maximum in this second derivative
    # this is where the curve 'drops off'
    min_modul_ix = dif2.sort_values(ascending=False).index[0]
    min_modul = ld[min_modul_ix]

    # return as a namedtuple
    ModulationLevel = namedtuple('ModulationLevel', ['median', 'minimum'])
    ml = ModulationLevel(median_modul, min_modul)
    return ml
