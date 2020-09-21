""" TODO docstring """

# pylint: disable=E1101
# SOURCE: https://github.com/PyCQA/pylint/issues/2289

import numpy as np
import pandas as pd
from matplotlib import cm, pyplot, style
from matplotlib.dates import date2num
from matplotlib.dates import HourLocator, AutoDateLocator, DateFormatter
from matplotlib.colors import LogNorm


def plot_style():
    """ TODO docstring """

    style.use('seaborn-talk')
    style.use('seaborn-whitegrid')
    style.use('seaborn-deep')

    pyplot.rcParams['figure.figsize'] = 16, 6
    pyplot.rcParams['legend.facecolor'] = "#ffffff"
    pyplot.rcParams['legend.frameon'] = True
    pyplot.rcParams['legend.framealpha'] = 1


def carpet(time_series, options=None):
    """
    Draw a carpet plot of a pandas time_series.

    The carpet plot reads like a letter. Every day one line is added to the
    bottom of the figure, minute for minute moving from left (morning) to right
    (evening).
    The color denotes the level of consumption and is scaled logarithmically.
    If vmin and vmax are not provided as inputs, the minimum and maximum of the
    colorbar represent the minimum and maximum of the (resampled) time_series.

    Parameters
    ----------
    time_series :
        pandas.Series
    vmin, vmax :
        If not None, either or both of these values determine the range of the z axis.
        If None, the range is given by the minimum and/or maximum of the (resampled) time_series.
    zlabel, title :
        If not None, these determine the labels of z axis and/or title. If None, the name of
        the time_series is used if defined.
    cmap :
        matplotlib.cm instance, default coolwarm

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from opengrid.library import plotting
    >>> pyplot = plotting.plot_style()
    >>> index = pd.date_range('2015-1-1','2015-12-31',freq='h')
    >>> ser = pd.Series(np.random.normal(size=len(index)), index=index, name='abc')
    >>> im = plotting.carpet(ser)
    """

    # data preparation
    if time_series.dropna().empty:
        print('skipped - no data')
        return None

    options = options if options else {}

    time_series = time_series.resample('15min').interpolate()
    time_series_name = time_series.name if "name" in time_series else ""

    # convert to dataframe with date as index and time as columns by
    # first replacing the index by a MultiIndex
    mpldatetimes = date2num(time_series.index.to_pydatetime())
    time_series.index = pd.MultiIndex.from_arrays([np.floor(mpldatetimes),
                                                   2 + mpldatetimes % 1])
    # and then unstacking the second index level to columns
    data_frame = time_series.unstack()

    # data plotting

    _fig, axes = pyplot.subplots()

    vmin = options.pop('vmin', time_series[time_series > 0].min())
    vmin = max(0.1, vmin)

    vmax = options.pop('vmax', time_series.quantile(.999))
    vmax = max(vmin, vmax)

    extent = [data_frame.columns[0],
              data_frame.columns[-1],
              data_frame.index[-1] + 0.5,   # + 0.5 to align date ticks
              data_frame.index[0] - 0.5]    # - 0.5 to align date ticks

    axes_image = pyplot.imshow(X=data_frame,
                               vmin=vmin,
                               vmax=vmax,
                               extent=extent,
                               cmap=options.pop('cmap', cm.coolwarm),
                               aspect=options.pop('aspect', 'auto'),
                               norm=options.pop('norm', LogNorm()),
                               interpolation=options.pop('interpolation', 'nearest'))

    # figure formatting
    # x axis
    axes.xaxis_date()
    axes.xaxis.set_major_locator(HourLocator(interval=2))
    axes.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    axes.xaxis.grid(True)
    pyplot.xlabel('UTC Time')

    # y axis
    axes.yaxis_date()

    # AutoDateLocator is not suited in case few data is available
    axes.yaxis.set_major_locator(AutoDateLocator())
    axes.yaxis.set_major_formatter(DateFormatter("%a, %d %b %Y"))

    # plot colorbar
    colorbar_label = options.pop('zlabel', time_series_name)
    colorbar_ticks = np.logspace(start=np.log10(vmin),
                                 stop=np.log10(vmax),
                                 num=11,
                                 endpoint=True)
    colorbar = pyplot.colorbar(format='%.0f',
                               ticks=colorbar_ticks)
    colorbar.set_label(colorbar_label)

    # plot title
    pyplot.title("Carpet plot - %s" % time_series_name)

    return axes_image


def boxplot(data_frame, plot_mean=False, plot_ids=None, title="", labels: tuple = ("x", "y")):
    """
    Plot boxplots

    Plot the boxplots of a dataframe in time

    Parameters
    ----------
    data_frame: Pandas Dataframe
        Every collumn is a time_series
    plot_mean: bool
        Wether or not to plot the means
    plot_ids: [str]
        List of id's to plot

    Returns
    -------
    matplotlib figure
    """

    data_frame = data_frame.applymap(float)
    description = data_frame.apply(func=pd.DataFrame.describe,
                                   axis=1)

    # plot
    plot_style()

    pyplot.boxplot(x=data_frame)
    # pyplot.setp(bp['boxes'], color='black')
    # pyplot.setp(bp['whiskers'], color='black')
    if plot_ids:
        for plot_id in (x if x in data_frame.columns else None for x in plot_ids):
            pyplot.scatter(x=range(1, len(data_frame) + 1),
                           y=data_frame[plot_id],
                           label=str(plot_id))

    if plot_mean:
        pyplot.scatter(x=range(1, len(data_frame) + 1),
                       y=description['mean'],
                       label="Mean",
                       color='k',
                       s=30,
                       marker='+')

    axes = pyplot.gca()
    axes.set_xticks(data_frame.index)

    pyplot.legend()
    pyplot.title(title)
    pyplot.xlabel(labels[0])
    pyplot.ylabel(labels[1])

    return pyplot.gcf()
