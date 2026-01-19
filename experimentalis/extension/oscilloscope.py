"""
This extension module is used for handling Oscilloscopes (and their data).
"""

import numpy as np
from experimentalis.dataset import Dataset
from experimentalis.plotting import GraphingOptions

"""
A default Voltage vs. Time graphing options instance for simplicity.
"""
VOLTAGE_VERSUS_TIME_GRAPH_OPTIONS = GraphingOptions(
    x_label='Time',
    y_label='Voltage',
    x_units='s',
    y_units='V'
)

def load_channel(filename: str):
    """
    Loads the data of a singular oscilloscope channel from a CSV file.

    :param filename: The filepath of the dataset CSV file.
    :type filename: str

    :returns: A dataset of the loaded channel data.
    :rtype: Dataset
    """
    data = np.loadtxt(filename,
                      delimiter=',',
                      comments='#',
                      usecols=(3,4),
                      skiprows=1)
    xvalues = data[:,0]
    yvalues = data[:,1]
    
    dx = np.zeros_like(xvalues)
    dy = np.zeros_like(yvalues)
    
    return Dataset(x=xvalues, y=yvalues, dx=dx, dy=dy)

def plot_channels(ch1: Dataset,
                  ch2: Dataset,
                  graphing_options: GraphingOptions):
    """
    Plots two simultaneous Oscilloscope channels for analysis.

    :param ch1: The first channel's data.
    :param ch2: The second channel's data.
    :type ch1: Dataset
    :type ch2: Dataset

    :param graphing_options: The graphing options for the plot.
    :type graphing_options: GraphingOptions
    """
    plt.figure()
    plt.title('Channels 1 and 2')
    graphing_options.plot_data(ch1.x, ch1.y, ch1.dx, ch1.dy, label='Channel 1', color='lightblue')
    graphing_options.plot_data(ch2.x, ch2.y, ch2.dx, ch2.dy, label='Channel 2', color='orange')
    graphing_options.set_labels()
    plt.legend()
    plt.show()

def plot_channel_lissajous(ch1: Dataset,
                           ch2: Dataset,
                           graphing_options: GraphingOptions):
    """
    Produces a `Lissajous curve <https://en.wikipedia.org/wiki/Lissajous_curve>`__ for the two channels.

    :param ch1: The first channel's data.
    :param ch2: The second channel's data.
    :type ch1: Dataset
    :type ch2: Dataset

    :param graphing_options: The graphing options for the plot.
    :type graphing_options: GraphingOptions
    """
    plt.figure()
    plt.title('Channels 1 and 2 Lissajou')
    plt.errorbar(ch1.y, ch2.y, xerr=ch1.dy, yerr=ch2.dy, marker='.', linestyle='-')
    graphing_options.set_labels()
    plt.show()    

def load_raw_oscilloscope_data(filename: str):
    """
    Loads raw oscilloscope data to a dataset.

    :param filename: The filepath of the raw oscilloscope data in CSV format.
    :type filename: str
    
    :returns: A dataset containing the raw `x` and `y` values.
    :rtype: Dataset
    """
    data = np.loadtxt(filename,
                      delimiter=',',
                      comments='#',
                      usecols=(3,4),
                      skiprows=1)
    xvalues = data[:,0]
    yvalues = data[:,1]
    
    return Dataset(
        x = xvalues,
        dx = np.zeros_like(xvalues),
        y = yvalues,
        dy = np.zeros_like(yvalues)
    )
