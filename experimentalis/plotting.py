"""
The plotting module is used for automatically managing visuals, like a sort
of wrapper for MatPlotLib directives. The most core object here is
`GraphingOptions`, which is utilized to create standardized visuals through
having a standard settings for all graphs produced.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from matplotlib import pyplot as plt
from matplotlib import colors
import io
from PIL import Image
from abc import ABC, abstractmethod
from .dataset import Dataset

@dataclass
class GraphingOptions:
    """
    A standardized structure for creating visuals with the same layout and
    style, such that each plot doesn't need to be configured manually.

    :param x_label: The label along the `x`-axis.
    :param y_label: The label along the `y`-axis.
    :param x_units: The units of `x` (optional).
    :param y_units: The units of `y` (optional).

    :type x_label: str
    :type y_label: str
    :type x_units: str
    :type y_units: str

    :param data_round: The current round of data collection, used for
    automatic title generation.
    :type data_round: int
    """
    x_label: str = ''
    y_label: str = ''
    x_units: str = None
    y_units: str = None
    
    data_marker:      str   = '.'
    data_marker_size: int   = 2
    data_linestyle:   str   = ''
    data_alpha:       float = 0.80
    data_color:       str   = 'C0'
    
    model_marker:     str   = ''
    model_linestyle:  str   = '-'
    model_linewidth:  int   = 2
    model_alpha:      float = 1.0
    model_color:      str   = 'darkred' 
    
    data_round: int = 1

    def set_labels(self, xlabel=None, ylabel=None):
        """
        Automatically sets the labels for a the next figure based on
        whether or not the label names are configured to use units.

        :param xlabel: Optional label for the `x`-axis, otherwise defaults
        to the label previously set.
        :type xlabel: str
        :type ylabel: str
        """
        if self.x_units is not None:
            plt.xlabel(f"{self.x_label} ({self.x_units})")
        else:
            plt.xlabel(self.x_label)

        if self.y_units is not None:
            plt.ylabel(f"{self.y_label} ({self.y_units})")
        else:
            plt.ylabel(self.y_label)
            
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
            
    def plot_individual_dataset(self, data: Dataset,
                                label: str = None, color: str = None):
        """
        Plots an individual dataset to an active figure.

        :param data: The dataset to plot
        :param label: The label for this dataset
        :param color: The color for this plot

        :type data: Dataset
        :type label: str
        :type color: str
        """
        base_color = color if color is not None else self.data_color
        base_rgb = colors.to_rgb(base_color)

        plt.errorbar(
            data.x,
            data.y,
            xerr=data.dx,
            yerr=data.dy,
            marker=self.data_marker,
            markersize=self.data_marker_size,
            linestyle=self.data_linestyle,

            # opaque markers
            color=base_color,

            # translucent error bars
            ecolor=(*base_rgb, 0.25),

            elinewidth=1.0,
            capsize=3,
            label=label,
        )

    def plot_datasets(self,
                      datasets: list[Dataset],
                      labels: list[str] = None,
                      colors: list[str] = None):
        """
        Plots multiple datasets to the image, with optional, but highly
        recommended custom labels and colors.

        :param datasets: A list of the datasets to plot.
        :param labels: A list of the labels for each dataset.
        :param colors: A list of the colors for each series.

        :type datasets: list[Dataset]
        :type labels: list[str]
        :type colors: list[str]
        """
        plt.figure()
        plt.title(graphing_options.default_title())

        n = len(datasets)
        if labels is None:
            labels = [None] * n
        if colors is None:
            colors = [None] * n
        
        for dataset, label, color in zip(datasets, labels, colors):
            graphing_options.plot_data(dataset, label, color)
            
        graphing_options.set_labels()
        plt.show()
    
    def plot_model(self, model_x: NDArray, model_y: NDArray):
        """
        Plots a model curve to an active plot.

        :param model_x: The `x`-values for the model.
        :param model_y: The `y`-values for the model.
        :type model_x: NDArray
        :type model_y: NDArray
        """
        plt.plot(model_x,
                 model_y, 
                 marker    = self.model_marker, 
                 linestyle = self.model_linestyle, 
                 linewidth = self.model_linewidth,
                 alpha     = self.model_alpha,
                 color     = self.model_color,
                 label     = f'Fit')
    
    def plot_residuals(self,
                       x: NDArray,
                       residuals: NDArray,
                       y_uncert: NDArray):
        """
        Produces a residuals plot for a model.

        :param x: The `x`-values of the data.
        :param residuals: The residuals from the model.
        :param y_uncert: The uncertainty in each residual/datapoint.

        :type x: NDArray
        :type residuals: NDArray
        :type y_uncert: NDArray
        """

        # Automatically set the title and labels
        plt.title("Residuals")
        if self.y_units is not None:
            plt.ylabel(f"Residual $y-y_{{fit}}$ [{self.y_units}]")
        else:
            plt.ylabel(r"Residual $y-y_{fit}$")

        # Plots the Residuals
        plt.errorbar(x, residuals, yerr=y_uncert, 
                     marker     = self.data_marker,
                     markersize = self.data_marker_size,
                     linestyle  = self.data_linestyle,
                     alpha      = self.data_alpha,
                     color      = self.data_color,
                     label      = "Residuals")

        # Plots the model line at y=0
        plt.axhline(y=0, 
                    marker    = self.model_marker, 
                    linestyle = self.model_linestyle, 
                    linewidth = self.model_linewidth,
                    alpha     = self.model_alpha,
                    color     = self.model_color, 
                    label     = f'Fit')

    @staticmethod
    def save_graph_and_close():
        """
        Saves the active figure to a png-representation variable for
        ease of display or saving later.

        .. code-block:: python

            from matplotlib import pyplot as plt
            from IPython import display

            graphing_options = GraphingOptions(...)
        
            plt.figure()
            
            # some plotting code ..

            figure = graphing_options.save_graph_and_close()

            # other things

            display(figure)
        """
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        graph = Image.open(buf)
        plt.close()
        return graph
        
    def default_title(self):
        """
        Automatically generates a default title for data plotted to a
        figure, which is just the dependent and independent variables
        alongside the round number.

        :returns: The default title
        :rtype: str
        """
        return f'{self.y_label} vs. {self.x_label}, Round {self.data_round}'

