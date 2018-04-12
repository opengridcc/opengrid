import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


def plot_style():
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    matplotlib.style.use('seaborn-talk')
    matplotlib.style.use('seaborn-whitegrid')
    matplotlib.style.use('seaborn-deep')

    plt.rcParams['figure.figsize'] = 16, 6

    # To overrule the legend style
    plt.rcParams['legend.facecolor'] = "#ffffff"
    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 1

    return plt


def boxplot(df, plot_mean=False, plot_ids=None):
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
    description = df.apply(pd.DataFrame.describe, axis=1)

    # plot
    plt = plot_style()

    df.index = df.index.map(lambda x: x.strftime('%b'))

    df = df.T

    fig, ax = plt.subplots()
    axes, bp = df.boxplot(ax=ax, return_type='both')
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='black')

    for id in plot_ids:
        ax.scatter(x=axes.get_xticks(), y=df.loc[id], label=str(id))

    if plot_mean:
        ax.scatter(x=axes.get_xticks(), y=description['mean'], label="Mean", color='k', s=30, marker='+')

    plt.xticks(rotation=45)

    ax.legend()

    return fig

