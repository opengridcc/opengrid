# -*- coding: utf-8 -*-
"""
Weather data functionality.

Try to write all methods such that they take a dataframe as input
and return a dataframe.
"""
import pandas as pd
from opengrid.library.exceptions import *


def calculate_temperature_equivalent(temperatures):
    """
    Calculates the temperature equivalent from a series of average daily temperatures
    according to the formula: 0.6 * tempDay0 + 0.3 * tempDay-1 + 0.1 * tempDay-2

    Parameters
    ----------
    series : Pandas Series

    Returns
    -------
    Pandas Series
    """

    ret = 0.6*temperatures + 0.3*temperatures.shift(1) + 0.1*temperatures.shift(2)
    ret.name = 'temp_equivalent'
    return ret


def _calculate_degree_days(temperature_equivalent, base_temperature, cooling=False):
    """
    Calculates degree days, starting with a series of temperature equivalent values

    Parameters
    ----------
    temperature_equivalent : Pandas Series
    base_temperature : float
    cooling : bool
        Set True if you want cooling degree days instead of heating degree days

    Returns
    -------
    Pandas Series called HDD_base_temperature for heating degree days or
    CDD_base_temperature for cooling degree days.
    """

    if cooling:
        ret = temperature_equivalent - base_temperature
    else:
        ret = base_temperature - temperature_equivalent

    # degree days cannot be negative
    ret[ret < 0] = 0

    prefix = 'CDD' if cooling else 'HDD'
    ret.name = '{}_{}'.format(prefix, base_temperature)

    return ret


def compute_degree_days(ts, heating_base_temperatures, cooling_base_temperatures):
    """
    Compute degree-days for heating and/or cooling

    Parameters
    ----------
    ts : pandas.Series
        Contains ambient (outside) temperature. Series name (ts.name) does not matter.
    heating_base_temperatures: list
        For each base temperature the heating degree-days will be computed
    cooling_base_temperatures: list
        For each base temperature the cooling degree-days will be computed

    Returns
    -------
    df: pandas.DataFrame with DAILY resolution and the following columns:
        temp_equivalent and columns HDD_baseT and CDD_baseT for each of the given base temperatures.
    """

    # verify the sampling rate: should be at least daily.
    mean_sampling_rate = (ts.index[-1] - ts.index[0]).total_seconds()/(len(ts)-1)
    if int(mean_sampling_rate/86400.) > 1:
        raise UnexpectedSamplingRate("The sampling rate should be daily or shorter but found sampling rate: {}s".format(mean_sampling_rate))

    ts_day = ts.resample(rule='D').mean()
    df = pd.DataFrame(calculate_temperature_equivalent(ts_day))

    for base in heating_base_temperatures:
        df = pd.concat([df, _calculate_degree_days(temperature_equivalent=df['temp_equivalent'], base_temperature=base)], axis=1)

    for base in cooling_base_temperatures:
        df = pd.concat([df, _calculate_degree_days(temperature_equivalent=df['temp_equivalent'], base_temperature=base, cooling=True)],
                       axis=1)

    return df



