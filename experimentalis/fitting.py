"""
The fitting module is for actually performing fits to data using the
previously defined fit models to estimate model parameters.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from dataclasses import dataclass
from PIL import Image
from typing import Callable

from .plotting import GraphingOptions
from .models import Model
from .data import Dataset

@dataclass
class FitModelResult:
    """
    A result structure for collecting and displaying fit results visually.
    Produced automatically after performing an ``autofit``.

    :ivar initial_guess_graph: A plot of the model pre-optimization, using the initial parameter guesses (if provided).
    :vartype initial_guess_graph: Image

    :ivar initial_guess_residuals_graph: A plot of the residuals of the initial values.
    :vartype initial_guess_residuals_graph: Image

    :ivar autofit_graph: A plot of the model post-optimization.
    :vartype autofit_graph: Image

    :ivar autofit_residuals_graph: A plot of the post-optimization model residuals.
    :vartype autofit_residuals_graph: Image

    :ivar chi2: The goodness-of-fit of the model post-optimization.
    :vartype chi2: float
    """
    initial_guess_graph:           Image = None
    initial_guess_residuals_graph: Image = None
    autofit_graph:                 Image = None
    autofit_residuals_graph:       Image = None

    chi2:              float = None
    covariance_matrix: NDArray = None
    
def print_results(model: Model,
                  results: FitModelResult,
                  print_cov: bool = False,
                  units: list[str] = None):
    """
    Prints the results of a fitting in a tabulated format, with the optional
    inclusion of the covariance matrix for the parameters.

    :param model: The fit model, used to access the optimized parameters.
    :type model: Model

    :param results: The results of the fitting. Used to access the covariance matrix and the goodness-of-fit.
    :type results: FitModelResults

    :param print_cov: Whether or not to include the covariance matrix.
    :type print_cov: bool

    :param units: The (optional) ordered list of units for each parameter.
    :type units: list[str]
    """
    display(model.tabulate(units=units))
    display(Markdown(f"Goodness of Fit: $\\chi$² = {results.chi2:.3f}"))

    if print_cov:
        display(Markdown("**Covariance Values**"))
        for i in range(len(model.param_names)):
            for j in range(i + 1, len(model.param_names)):
                display(Markdown(
                    f"- **{model.param_names[i]} & {model.param_names[j]}**: "
                    f"{results.covariance_matrix[i, j]:.3e}"
                ))

def calculate_chi_squared(model: Model, dataset: Dataset):
    """
    Calculates the goodness-of-fit :math:`\\chi^2` for a given model
    post-fitting via the equation

    .. math::
        \\chi^2 = \\frac{1}{N} \\sum_{i=1}^{N} \\left( \\frac{y_i - f(x_i)}{dy} \\right)^2

    given model :math:`f` and dataset :math:`D_i=(x_i,y_i)`.
    
    :param model: The model post-optimization.
    :type model: Model

    :param dataset: The dataset to compare against.
    :type dataset: Dataset

    :returns: The goodness-of-fit :math:`\\chi^2`.
    """
    dof = len(data.x) - len(model.param_values)
    residuals = data.y - fit_function(data.x, *model.param_values)
    return np.sum((residuals/data.dy) ** 2) / dof

def autofit(data: Dataset,
            model: Model,
            graphing_options: GraphingOptions):
    """
    Optimizes a model to fit a dataset (editing the model in-place), and
    plots the results to a ``FitModelResult`` object.

    :param data: The dataset to fit to.
    :param model: The model.
    :param graphing_options: The graphing options for plotting.

    :type data: Dataset
    :type model: Model
    :type graphing_options: GraphingOptions

    :returns: The results of the optimization.
    :rtype: FitModelResult
    """
    results      = FitModelResult()
    fit_function = model.fit_function
    guesses      = model.values()
    
    # Theoretical x and y values for the sake of plotting
    guess_model_x = np.linspace(min(data.x), max(data.x), 500)
    guess_model_y  = fit_function(guess_model_x, *guesses)
    
    plt.figure()
    graphing_options.set_labels()
    plt.title('Initial Parameter Guess')
    graphing_options.plot_data(data.x, data.y, data.dx, data.dy, label='Measured Data')
    graphing_options.plot_model(guess_model_x, guess_model_y)
    plt.legend(loc="best", numpoints=1)
    results.initial_guess_graph = graphing_options.save_graph_and_close()

    # Residuals
    guess_y_fit = fit_function(data.x, *guesses)
    guess_residuals = data.y - guess_y_fit

    # Plot the residuals
    plt.figure()
    graphing_options.set_labels()
    plt.title("Residuals of Initial Parameter Guess")
    graphing_options.plot_residuals(data.x, guess_residuals, data.dy)
    results.initial_guess_residuals_graph = graphing_options.save_graph_and_close()

    # Perform the fit
    kwargs = dict(
        p0 = guesses,
        absolute_sigma = True,
        maxfev = int(1e5),
        sigma = data.dy
    )
    
    if model.has_bounds():
        kwargs["bounds"] = model.bounds()
    
    fit_params, fit_cov = curve_fit(fit_function, data.x, data.y, **kwargs)
    fit_params_error = np.sqrt(np.diag(fit_cov))
    
    # Store the fit results
    model.update_fit_results(fit_params, fit_params_error)
    
    results.chi2 = calculate_chi_squared(model, data)
    results.covariance_matrix = fit_cov

    # Evaluate the Autofit
    
    model_x = np.linspace(min(data.x), max(data.x), len(data.x))
    model_y = fit_function(model_x, *fit_params)
    y_fit = fit_function(data.x, *fit_params)
    residuals = data.y - y_fit

    plt.figure()
    graphing_options.set_labels()
    plt.title('Best Fit of Function to Data')
    graphing_options.plot_data(data.x, data.y, data.dx, data.dy, label='Measured Data');
    graphing_options.plot_model(model_x, model_y);
    plt.legend(loc='best',numpoints=1)
    results.autofit_graph = graphing_options.save_graph_and_close()
    
    # The residuals plot
    plt.figure()
    graphing_options.set_labels()
    graphing_options.plot_residuals(data.x, residuals, data.dy)
    results.autofit_residuals_graph = graphing_options.save_graph_and_close()

    return results
