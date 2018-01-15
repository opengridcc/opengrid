import matplotlib.pyplot as plt
import matplotlib


def plot_style():
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except NameError:
        pass

    matplotlib.style.use('seaborn-talk')
    matplotlib.style.use('seaborn-whitegrid')
    matplotlib.style.use('seaborn-deep')

    plt.rcParams['figure.figsize'] = 16, 6
    return plt
