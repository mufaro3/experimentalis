from dataclasses import dataclass
from numpy.typing import NDArray

@dataclass
class Dataset:
    """
    Datasets are tracker objects for two-dimensional coupled measurement,
    i.e., some independent measurement `x` and its uncertainty `dx` alongside
    a dependent measurement `y` on `x` and its uncertainty `dy`.

    For example, a Dataset can typically be used for any one-dimensional
    time series, like stock prices, temperature, or precipitation chances.
    For the case of stock prices, `x` would be the time, `dx` would be the
    uncertainty in each clock reading (so, say, the instrumental lag between
    a measurement and the intended time delta), `y` would be the stock
    prices, and `dy` would be the uncertainty in the stock reading (once
    again due to lag or other issues).
    """
    x:  NDArray
    dx: NDArray
    y:  NDArray
    dy: NDArray
