import numpy as np
from typing import Callable, Optional, Tuple
from numpy.typing import NDArray
from experimentalis.models import Model
from abc import ABC, abstractmethod

# no CUDA usage, super unnecessary and not cross platform
import tensorflow as tf

class IntegratedItoSDEModel(Model):
    r"""
    This is a base class for implementing various superclass models in the family of stochastic PDEs,
    as solved via `Ito's lemma <https://en.wikipedia.org/wiki/It%C3%B4's_lemma>`__, following the
    equation

    .. math::

        dS_t = \mu(S_t, t) dt + \sigma(S_t, t) dB_t
    """
    def __init__(
            self,
            integrated_function: Callable[[NDArray, NDArray, NDArray], NDArray],
            initial_value: float,
            param_names: list[str],
            param_values: NDArray,
            param_bounds: Optional[Tuple[NDArray, NDArray]] = None
    ):
        super().__init__(
            fit_function=self.evaluate,
            param_names=param_names,
            param_values=param_values,
            param_uncerts=np.full_like(param_values, -1.0),
            param_bounds=param_bounds
        )
        
        self.initial_value = initial_value
        self.integrated_function = integrated_function
        
    def evaluate(self, t: NDArray, *params: float | NDArray) -> NDArray:
        """
        Evaluates the SDE integrated solution with the given parameters.

        :param t: Array of times to evaluate on
        :type  t: NDArray
        
        :param params: Parameter values
        :type  params: float or NDArray 
        """
        return self.integrated_function(
            np.asarray(t),
            self.initial_value,
            *params
        )

def gbm_function(t: NDArray, initial_value: float, drift: float, volatility: float):
    """Implementation of the GBM equation

    .. math::

        S_t = S_0 \\exp\\left\\{\\sigma B_t + \\left( \\mu - \\frac{\\sigma^2}{2} \\right) t\\right\\}

    :param t: The time values to evaluate on
    :type  t: NDArray

    :param initial_value: :math:`S_0` The initial value
    :type  initial_value: float

    :param drift: :math:`\\mu` The percentage drift (predictability).
    :type  drfit: float

    :param volatility: :math:`\\sigma` The percentage volatility of the dataset.
    :type  volatility: float

    :param drift_bounds: Optional bounds on the geometric drift.
    :type  drift_bounds: tuple[float, float] or None

    :param volatility_bounds: Optional bounds on the volatility.
    :type  volatility_bounds: tuple[float, float] or None
    """
    wiener = np.random.normal(0, np.sqrt(np.diff(np.insert(t, 0, 0))), size=t.shape).cumsum()
    return initial_value * np.exp(volatility * wiener + (drift - 0.5 * volatility ** 2) * t)

    
class GBM(IntegratedItoSDEModel):
    """
    This is a very generic stochastic differential equation (SDE) model in 1D using
    `Geometric Brownian Motion <https://en.wikipedia.org/wiki/Geometric_Brownian_motion>`__.
    (GBM) in integrated form.

    Example:

    .. code-block:: python

        # some time-series dataset with inherent randomness
        dataset = Dataset(...)

        ...

        model = GBM(
            initial_value = prices[0],
            drift = 1.01e-3,             # very small drift, i.e., predictibility
            drift_bounds = (0,0.1),      # depending on data, drift is usually small
            volatility = 2e-2,           # higher volatility
            volatility_bounds = (0,1)    # volatility can be higher, but this makes sense for the data
        )

        result = autofit(dataset, model, graphing_options=g_opts)
        ...    
    """
    def __init__(self, initial_value: float, drift: float, volatility: float,
                 drift_bounds: tuple[float, float] = (-1.0,1.0),
                 volatility_bounds: tuple[float, float] = (1e-6,2.0)):

        lower_bounds = np.array([drift_bounds[0], volatility_bounds[0]])
        upper_bounds = np.array([drift_bounds[1], volatility_bounds[1]])
        
        super().__init__(
            integrated_function=gbm_function,
            initial_value=initial_value,
            param_names=['Drift', 'Volatility'],
            param_values=[drift, volatility],
            param_bounds=(lower_bounds, upper_bounds)
        )

class DriftNet(tf.keras.Model):
    """
    This is a drift network form for automatically learning the stochastic drift :math:`\\mu` from the data
    rather than manually guessing, as a means to curb the local-minima issue resulting from the naive model.

    :param hidden_units: The number of hidden units (weight + bias) to use in the model, effectively a measure of the model's complexity.
    :type  hidden_units: int
    """
    def __init__(self, hidden_units: int, eps: float = 1e-3):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(hidden_units, activation='tanh',
                                        kernel_regularizer=tf.keras.regularizers.l2(eps))
        self.d2 = tf.keras.layers.Dense(1)

    def call(self, t):
        t = tf.reshape(t, (-1,1))
        return tf.squeeze(self.d2(self.d1(t)))

class LearnedDriftGBM(Model):
    """
    This is another implementation of the integrated Geometric Brownian Motion model for a stochastic
    differential dataset, but in this case, the drift parameter :math:`\\mu` is learned from the data
    using machine learning, and only the volatility :math:`\\sigma` is set manually.
    """
    def __init__(self, initial_value: float, volatility: float, hidden_units: int = 8,
                 volatility_bounds: tuple[float, float] = (1e-6,2.0),
                 weight_init_scale: float = 0.1, use_cpu=False):
        self.initial_value = initial_value
        self.drift_model = DriftNet(hidden_units)
        self.drift_model(tf.constant([[0.0]], dtype=tf.float32))
        
        # flatten the model weights
        self.weight_shapes = [ w.shape for w in self.drift_model.weights ]
        flat_weights = np.concatenate([
            (np.random.randn(*w.shape).ravel() * weight_init_scale)
            for w in self.drift_model.weights
        ])
        num_weights = flat_weights.size
        
        # set the parameter bounds
        volatility_min, volatility_max = volatility_bounds
        lower_bounds = np.concatenate([
            np.array([volatility_min]),
            np.full(num_weights, -np.inf)
        ])
        upper_bounds = np.concatenate([
            np.array([volatility_max]),
            np.full(num_weights, np.inf)
        ])
        
        super().__init__(
            fit_function  = self.evaluate,
            param_names   = [ 'Volatility' ] + [ f'Weight {i}' for i in range(num_weights) ],
            param_values  = [ volatility, *flat_weights ],
            param_uncerts = np.full(num_weights + 1, -1.0),
            param_bounds  = (lower_bounds, upper_bounds)
        )

    def unflatten_weights(self, flat: NDArray | float):
        """
        Unflattens the weights to obtain new weights for the model.

        :param flat: the flattened weights.
        :type  flat: NDArray or float
        :meta private:
        """
        rebuilt = []
        index = 0
        for shape in self.weight_shapes:
            size = np.prod(shape)
            rebuilt.append(tf.reshape(flat[index:index + size], shape))
            index += size
        return rebuilt

    def evaluate(self, t: float | NDArray, volatility: float, *weights: float | NDArray):
        """
        Simultaneously trains the model and evaluates it on the present data.

        :param t: The time data to evaluate at.
        :param volatility: The volatility of the data.
        :param weights: The current weights of the data.

        :type t: float or NDArray
        :type volatility: float
        :type weights: NDArray or float
        """
        t = np.atleast_1d(t)
        
        # inject weights into the model
        new_weights = self.unflatten_weights(weights)
        for weight_slot, new_weight in zip(self.drift_model.weights, new_weights):
            weight_slot.assign(new_weight)
        
        # compute drift
        drift = self.drift_model(tf.constant(t, dtype=tf.float32)).numpy()
        
        return gbm_function(t, self.initial_value, drift, volatility) 
