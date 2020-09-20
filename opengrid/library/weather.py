# -*- coding: utf-8 -*-
"""
Weather data functionality.

Try to write all methods such that they take a dataframe as input
and return a dataframe.
"""

import pandas as pd

from opengrid.library.exceptions import UnexpectedSamplingRate


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

    ret = 0.6*temperatures + 0.3 * \
        temperatures.shift(1) + 0.1*temperatures.shift(2)
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


def compute_degree_days(time_series, heating_base_temperatures, cooling_base_temperatures):
    """
    Compute degree-days for heating and/or cooling

    Parameters
    ----------
    time_series : pandas.Series
        Contains ambient (outside) temperature. Series name (time_series.name) does not matter.
    heating_base_temperatures: list
        For each base temperature the heating degree-days will be computed
    cooling_base_temperatures: list
        For each base temperature the cooling degree-days will be computed

    Returns
    -------
    data_frame: pandas.DataFrame with DAILY resolution and the following columns:
        temp_equivalent and columns HDD_baseT and CDD_baseT for each of the given base temperatures.
    """

    # verify the sampling rate: should be at least daily.
    mean_sampling_rate = (
        time_series.index[-1] - time_series.index[0]).total_seconds()/(len(time_series)-1)
    if int(mean_sampling_rate/86400.) > 1:
        raise UnexpectedSamplingRate(
            "Should be daily at most. Found %s" % mean_sampling_rate)

    time_series_day = time_series.resample(rule='D').mean()
    data_frame = pd.DataFrame(
        calculate_temperature_equivalent(time_series_day))

    for base in heating_base_temperatures:
        data_frame = pd.concat([data_frame, _calculate_degree_days(
            temperature_equivalent=data_frame['temp_equivalent'], base_temperature=base)], axis=1)

    for base in cooling_base_temperatures:
        data_frame = pd.concat(objs=[data_frame,
                                     _calculate_degree_days(
                                         temperature_equivalent=data_frame['temp_equivalent'],
                                         base_temperature=base, cooling=True)
                                     ],
                               axis=1)

    return data_frame
