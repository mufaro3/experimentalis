"""
The basic modeling module for the Experimentalis library. This module
provides some simple generic usecase models and the base model class,
like linear, sinusoid, exponential fits, etc. Additional, usecase-specific
models can be found under `experimentalis.modules`, such as the RC and LRC
response curve models.
"""

import pandas as pd
import numpy as np
from numpy.typing import NDArray 
from typing import Callable, Sequence, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class Model:
    """
    Base class for a fitting model, consisting of the mathematical function
    to use for fitting (``fit_function``), and information about each of the
    model's parameters, like the names (``param_names``), values
    (``param_values``), uncertainties (``param_uncerts``), and the bounds
    to use when fitting the model to some data (``param_bounds``).
    """
    
    fit_function:  Callable = None
    param_names:   Sequence[str] = field(default_factory=list)
    param_values:  np.ndarray = field(default_factory=lambda: np.array([]))
    param_uncerts: np.ndarray = field(default_factory=lambda: np.array([]))
    param_bounds:  Optional[Tuple[NDArray, NDArray]] = None

    def values(self):
        """
        :returns: Parameter values.
        :rtype: Tuple[float]
        """
        return tuple(self.param_values)

    def uncertainties(self):
        """
        :returns: Parameter uncertainties.
        :rtype: Tuple[float]
        """
        return tuple(self.param_uncerts)

    def labels(self):
        """
        :returns: Parameter names.
        :rtype: Tuple[str]
        """
        return tuple(self.param_names)
    
    def has_bounds(self) -> bool:
        """
        :returns: Whether or not this model has any parameters bounds.
        :rtype: bool
        """
        return self.param_bounds is not None

    def bounds(self):
        """
        :returns: The bounds of the model, if it has any.
        :rtype: Optional[Tuple[NDArray, NDArray]]
        """
        if self.param_bounds is None:
            raise ValueError("Model has no parameter bounds")
        return self.param_bounds

    def update_fit_results(self, fit_params, fit_errors):
        """
        Sets the model's parameters to a fresh set of values and
        uncertainties, post-modelling.
        
        :param fit_params: New parameters for the model.
        :type fit_params: NDArray
        :param fit_errors: New uncertainties for the model.
        :type fit_errors: NDArray
        """
        self.param_values = np.array(fit_params)
        self.param_uncerts = np.array(fit_errors)
    
    def tabulate(self, units: list[str] = None):
        """
        Tabulates the results as a pandas dataframe to be easily displayed
        as a LaTeX table in a Jupyter cell.

        :param units: Optional units to include in the table.
        :type units: list[str]

        :returns: A table representation of the model's parameters.
        :rtype: pandas.DataFrame
        """
        def apply_units(label, i):
            if units is None or len(units) <= i:
                return label
            else:
                return f"{label} ({units[i]})"

        uncert = np.array(self.uncertainties())
        val = np.array(self.values())
            
        data = {
            "Measurement": [
                apply_units(label, i)
                for i, label in enumerate(self.labels())
            ],
            "Value": val,
            "Uncertainty": uncert,
            "Relative Uncertainty ($\\%$):": (uncert/val)*100
        }

        df = pd.DataFrame(data)
        return df

class CustomFitModel(Model):
    """
    A generic model class implementation for performing custom fits by
    manually specifying values.

    .. code-block:: python

        import numpy as np
        from experimentalis.models import CustomFitModel

        def decay_model(t, A, tau, C):
            return A * np.exp(-t / tau) + C

        initial_params = {
            "A": 1.0,
            "tau": 2.5,
            "C": 0.1
        }

        model = CustomFitModel(
            fit_function=decay_model,
            initial_params=initial_params
        )

        # Evaluate the model at some points
        t = np.linspace(0, 10, 100)
        y = model.fit_function(t, *model.param_values)
    
    :param fit_function: The mathematical function for this model.
    :type fit_function: Callable

    :param initial_params: The parameter names and starting values for this function.
    :type initial_params: dict

    :param param_bounds: Bounds for the parameter values to be used during model optimization (optional).
    :type param_bounds: Optional[Tuple[NDArray, NDArray]]
    """
    initial_params = None
    
    def __init__(self,
                 fit_function: Callable,
                 initial_params: dict,
                 param_bounds: Optional[Tuple[NDArray, NDArray]] = None):
        super().__init__(
            fit_function=fit_function,
            param_names=list(initial_params.keys()),
            param_values=np.array(list(initial_params.values())),
            param_uncerts=np.full(len(initial_params), -1.0),
            param_bounds=param_bounds
        )

class ExponentialOffsetModel(Model):
    """
    Exponential decay model with a vertical offset.

    .. math::

       f(x) = A \exp(-x/\tau) + C

    :param amplitude: Amplitude :math:`A`
    :param time_constant: Time Constant :math:`\\tau`
    :param offset: Vertical Offset :math:`C`

    :type amplitude: float
    :type time_constant: float
    :type offset: float
    """
    def __init__(self, amplitude, time_constant, offset):
        def fit_function(x, amplitude, time_constant, offset):
            return amplitude * np.exp(-x/time_constant) + offset
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Time Constant', 'Offset'],
            param_values=np.array([amplitude, time_constant, offset]),
            param_uncerts=np.array([-1, -1, -1])
        )
        
class SineModel(Model):
    """
    Default sinusoid model.

    .. math::

        f(x) = A \sin (2 \pi f x + \phi)

    :param amplitude: Amplitude :math:`A`
    :param frequency: Frequency :math:`f`
    :param phase: Phase :math:`\\phi`

    :type amplitude: float
    :type frequency: float
    :type phase: float
    """
    def __init__(self, amplitude, frequency, phase):
        def fit_function(x, amplitude, freq, phase):
            return amplitude * np.sin(2.0 * np.pi * freq * x + phase)
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Frequency', 'Phase'],
            param_values=np.array([amplitude, frequency, phase]),
            param_uncerts=np.array([-1, -1, -1])
        )

class OffsetSineModel(Model):
    """
    Sinusoid model with a vertical offset.

    .. math::

        f(x) = A \sin (2 \pi f x + \phi) + C

    :param amplitude: Amplitude :math:`A`
    :param frequency: Frequency :math:`f`
    :param phase: Phase :math:`\\phi`
    :param offset: Offset :math:`C`

    :type amplitude: float
    :type frequency: float
    :type phase: float
    :type offset: float
    """
    def __init__(self, amplitude, frequency, phase, offset):
        def fit_function(x, amplitude, freq, phase, offset):
            return amplitude * np.sin(2.0 * np.pi * freq * x + phase) + offset
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Frequency', 'Phase', 'Offset'],
            param_values=np.array([amplitude, frequency, phase, offset]),
            param_uncerts=np.array([-1, -1, -1, -1])
        )

class DampedHarmonicModel(Model):
    """
    The `Damped Harmoinc Model`_, used for fitting the time-domain response of a
    lightly-damped harmonic oscillator (or other sinusoid).

    .. _Damped Harmonic: https://openstax.org/books/university-physics-volume-1/pages/15-5-damped-oscillations#fs-id1167134928721

    
    .. math::

        f(t) = A \exp(-t/\\tau) \cos(2 \pi f_0 t + \phi)

    :param amplitude: Amplitude :math:`A`
    :param time_constant: Frequency :math:`\\tau`
    :param frequency: Phase :math:`f_0`
    :param phase: Offset :math:`\\phi`

    :type amplitude: float
    :type time_constant: float
    :type frequency: float
    :type phase: float
    """
    def __init__(self, amplitude, time_constant, frequency, phase):
        def fit_function(x, amplitude, time_constant, frequency, phase):
            return amplitude * np.exp(-x/time_constant) * \
                np.cos(2.0 * np.pi * frequency * x + phase)
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Amplitude', 'Damping/Time Constant', 'Frequency', 'Phase'],
            param_values=np.array([amplitude, time_constant, frequency, phase]),
            param_uncerts=np.array([-1, -1, -1, -1])
        )

class OffsetLinearModel(Model):
    """
    A two-parameter linear fit (slope and intercept).

    .. math::

        f(x) = mx + b

    :param slope: Slope :math:`m`
    :param intercept: Intercept :math:`b`
    
    :type slope: float
    :type intercept: float
    """
    def __init__(self, slope, intercept):
        def fit_function(x, slope, intercept):
            return slope * x + intercept
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Slope', 'Intercept'],
            param_values=np.array([slope, intercept]),
            param_uncerts=np.array([-1, -1])
        )

class LinearModel(Model):
    """
    A one-parameter linear fit.

    .. math::

        f(x) = mx

    :param slope: Slope :math:`m`
    :type slope: float
    """
    def __init__(self, slope):
        def fit_function(x, slope):
            return slope * x
        
        super().__init__(
            fit_function=fit_function,
            param_names=['Slope'],
            param_values=np.array([slope]),
            param_uncerts=np.array([-1])
        )

