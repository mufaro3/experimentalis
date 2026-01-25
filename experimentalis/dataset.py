from dataclasses import dataclass
from numpy.typing import NDArray
from functools import reduce
import warnings
import numpy as np

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

    def __init__(self, x: NDArray, y: NDArray, dx: NDArray = None, dy: NDArray = None):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

        if dx is None:
            dx = np.zeros_like(x)
            warnings.warn('No dx set, setting all values to zero.')
        if dy is None:
            dy = np.zeros_like(y)
            warnings.warn('No dy set, setting all values to zero.')

        self.dx = dx
        self.dy = dy
            
        def dim(array: NDArray):
            return len(array.shape)
            
        def one_dim(arrays: list[NDArray]):
            each = [ dim(a) == 1 for a in arrays ]
            return reduce(lambda x, y: x and y, each)
            
        if not one_dim([self.x, self.y, self.dx, self.dy]):
            print(f'All parameters must be one-dimensional: x ~ {dim(x)}, y ~ {dim(y)}, dx ~ {dim(dx)}, dy ~ {dim(dy)}')
        if self.x.size != self.y.size:
            raise ValueError(f'x has length {self.x.size} and y has length {self.y.size}, both must be equal')


    def size(self):
        """
        Returns the size of the dataset
        """
        return self.x.size
