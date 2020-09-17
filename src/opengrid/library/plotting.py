import os
import os
import numpy as np
import pandas as pd
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date, HourLocator, DayLocator, AutoDateLocator, DateFormatter
from matplotlib.colors import LogNorm


def plot_style():
    # matplotlib inline, only when jupyter notebook
    # try-except causes problems in Pycharm Console
    if 'JPY_PARENT_PID' in os.environ:
        get_ipython().run_line_magic('matplotlib', 'inline')

    matplotlib.style.use('seaborn-talk')
    matplotlib.style.use('seaborn-whitegrid')
    matplotlib.style.use('seaborn-deep')

    plt.rcParams['figure.figsize'] = 16, 6

    # To overrule the legend style
    plt.rcParams['legend.facecolor'] = "#ffffff"
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 1

    return plt


def carpet(timeseries, **kwargs):
    """
    Draw a carpet plot of a pandas timeseries.

    The carpet plot reads like a letter. Every day one line is added to the
    bottom of the figure, minute for minute moving from left (morning) to right
    (evening).
    The color denotes the level of consumption and is scaled logarithmically.
    If vmin and vmax are not provided as inputs, the minimum and maximum of the
    colorbar represent the minimum and maximum of the (resampled) timeseries.

    Parameters
    ----------
    timeseries : pandas.Series
    vmin, vmax : If not None, either or both of these values determine the range
    of the z axis. If None, the range is given by the minimum and/or maximum
    of the (resampled) timeseries.
    zlabel, title : If not None, these determine the labels of z axis and/or
    title. If None, the name of the timeseries is used if defined.
    cmap : matplotlib.cm instance, default coolwarm

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from opengrid.library import plotting
    >>> plt = plotting.plot_style()
    >>> index = pd.date_range('2015-1-1','2015-12-31',freq='h')
    >>> ser = pd.Series(np.random.normal(size=len(index)), index=index, name='abc')
    >>> im = plotting.carpet(ser)
    """

    # define optional input parameters
    cmap = kwargs.pop('cmap', cm.coolwarm)
    norm = kwargs.pop('norm', LogNorm())
    interpolation = kwargs.pop('interpolation', 'nearest')
    cblabel = kwargs.pop('zlabel', timeseries.name if timeseries.name else '')
    title = kwargs.pop('title', 'carpet plot: ' + timeseries.name if timeseries.name else '')

    # data preparation
    if timeseries.dropna().empty:
        print('skipped {} - no data'.format(title))
        return
    ts = timeseries.resample('15min').interpolate()
    vmin = max(0.1, kwargs.pop('vmin', ts[ts > 0].min()))
    vmax = max(vmin, kwargs.pop('vmax', ts.quantile(.999)))

    # convert to dataframe with date as index and time as columns by
    # first replacing the index by a MultiIndex
    mpldatetimes = date2num(ts.index.to_pydatetime())
    ts.index = pd.MultiIndex.from_arrays(
        [np.floor(mpldatetimes), 2 + mpldatetimes % 1])  # '2 +': matplotlib bug workaround.
    # and then unstacking the second index level to columns
    df = ts.unstack()

    # data plotting

    fig, ax = plt.subplots()
    # define the extent of the axes (remark the +- 0.5  for the y axis in order to obtain aligned date ticks)
    extent = [df.columns[0], df.columns[-1], df.index[-1] + 0.5, df.index[0] - 0.5]
    im = plt.imshow(df, vmin=vmin, vmax=vmax, extent=extent, cmap=cmap, aspect='auto', norm=norm,
                    interpolation=interpolation, **kwargs)

    # figure formatting

    # x axis
    ax.xaxis_date()
    ax.xaxis.set_major_locator(HourLocator(interval=2))
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax.xaxis.grid(True)
    plt.xlabel('UTC Time')

    # y axis
    ax.yaxis_date()
    dmin, dmax = ax.yaxis.get_data_interval()
    number_of_days = (num2date(dmax) - num2date(dmin)).days
    # AutoDateLocator is not suited in case few data is available
    if abs(number_of_days) <= 35:
        ax.yaxis.set_major_locator(DayLocator())
    else:
        ax.yaxis.set_major_locator(AutoDateLocator())
    ax.yaxis.set_major_formatter(DateFormatter("%a, %d %b %Y"))

    # plot colorbar
    cbticks = np.logspace(np.log10(vmin), np.log10(vmax), 11, endpoint=True)
    cb = plt.colorbar(format='%.0f', ticks=cbticks)
    cb.set_label(cblabel)

    # plot title
    plt.title(title)

    return im


def boxplot(df, plot_mean=False, plot_ids=None, title=None, xlabel=None, ylabel=None):
    """
    Plot boxplots

    Plot the boxplots of a dataframe in time

    Parameters
    ----------
    df: Pandas Dataframe
        Every collumn is a timeseries
    plot_mean: bool
        Wether or not to plot the means
    plot_ids: [str]
        List of id's to plot

    Returns
    -------
    matplotlib figure
    """

    df = df.applymap(float)
    description = df.apply(pd.DataFrame.describe, axis=1)

    # plot
    plt = plot_style()

    plt.boxplot(df)
    #plt.setp(bp['boxes'], color='black')
    #plt.setp(bp['whiskers'], color='black')
    if plot_ids is not None:
        for id in plot_ids:
            if id in df.columns:
                plt.scatter(x=range(1, len(df) + 1), y=df[id], label=str(id))

    if plot_mean:
        plt.scatter(x=range(1, len(df) + 1), y=description['mean'], label="Mean", color='k', s=30, marker='+')

    ax = plt.gca()
    ax.set_xticklabels(df.index)
    #plt.xticks(rotation=45)

    plt.legend()
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    return plt.gcf()

