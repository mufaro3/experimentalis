"""
Data module for the Experimentalis library.
"""

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
import warnings
from .plotting import GraphingOptions
from .dataset import Dataset

def dataset_apply_selector(dataset: Dataset, selector):
    """
    Applies a selector to all dimensions of the dataset simultaneously.
    These selectors can be filters, sorters, ranges, etc.

    .. code-block:: python

        import numpy as np
        from experimentalis.data import Dataset

        data = Dataset(
            x  = [ 1, 2, 3 ],
            y  = [ 2, 4, 6 ],
            dx = [ 1/7, 1/6, 1/6 ],
            dy = [ 1/5, 1/9, 1/5 ]
        )

        last_two_selector = slice(1, 3)
        is_odd_selector   = data.y % 2 != 0

        last_two_values = dataset_apply_selector(data, last_two_selector)
        odd_values = dataset_apply_selector(data, is_odd_selector)
    
    :param dataset: Dataset to be transformed
    :type dataset: Dataset

    :param selector: A NumPy-compatible selector. This can be a slice, an index array, or a boolean mask.
    :type selector: slice or numpy.ndarray

    :returns: A new dataset containing only the selected entries.
    :rtype: Dataset
    """
    return Dataset(
        x  = dataset.x[selector],
        dx = dataset.dx[selector],
        y  = dataset.y[selector],
        dy = dataset.dy[selector],
    )
    
def sort_dataset(dataset: Dataset):
    """
    Automatically sorts a dataset along the x-axis.

    :param dataset: Dataset to be sorted
    :type dataset: Dataset

    :returns: A new dataset of the original sorted along the x-axis.
    :rtype: Dataset
    """
    order = dataset.x.argsort()
    return dataset_apply_selector(dataset, order)

def shear_dataset(dataset: Dataset, n: int):
    """
    Autoamtically shears of the last `n` entries of the dataset.

    :param dataset: Dataset to be sheared.
    :type dataset: experimentalis.data.Dataset

    :param n: The number of elements to remove.
    :type n: int
    
    :returns: A new dataset with the last `n` elements removed.
    :rtype: Dataset
    """
    if n == 0:
        return dataset
    return dataset_apply_selector(dataset, slice(None, -n))

def trim_dataset(dataset: Dataset,
                 trim_range: tuple[int],
                 graphing_options: GraphingOptions = None,
                 plot: bool = False):
    """
    Trims a dataset (optionally with a visual helper) within a selected
    range. The trim range is the data designated to be kept, not the data
    to remove.

    :param dataset: Dataset to be trimmed.
    :type dataset: Dataset

    :param trim_range: The trimming range
    :type trim_range: (int, int)

    :returns: A copy of the original dataset containing only the trimmed data.
    :rtype: Dataset
    """
    if trim_range is None:
        trim_range = (0, len(dataset.x))
    trimmed = dataset_apply_selector(dataset, slice(*trim_range))
    
    if plot:
        plt.figure()
        plt.scatter(indices, dataset.y, marker='.')
        graphing_options.set_labels(xlabel='Index')
        mask = (indices >= trim_range[0]) & (indices <= trim_range[1])

        plt.fill_between(
            indices, min(dataset.y), max(dataset.y), 
            where = mask, 
            color='green', 
            alpha=0.1, 
            label='Trimmed Range'
        )

        plt.legend()
        plt.show()
        
    return trimmed

def pack_dataset(dataset: Dataset, packing_factor: int = 100):
    r"""
    Downsamples a dataset by averaging consecutive blocks of data (packing).

    Returns a new Dataset where consecutive blocks of size p are averaged. The uncertainty in :math:`y` is reduced accordingly. Implicitly ignores :math:`dx`, so the function automatically throws a warning for any dataset where :math:`dx \neq \vec{0}`.

    Mathematically, if our dataset :math:`D` is

    .. math::
        
        \vec{D} = (x_i, y_i, dx_i, dy_i), \quad i = 1, 2, \dots, N

    and we have packing factor :math:`p` (the block size), then for each
    packed point :math:`D'_j` for :math:`j \in 1, 2, \dots, N/p`,

    .. math::

        D'_j = (x'_j, y'_j, 0, dy'_j)

    where

    .. math::

        x'_j = \frac{1}{p} \sum_{k=0}^{p-1} x_{(j-1)p + k},
    
        y'_j = \frac{1}{p} \sum_{k=0}^{p-1} y_{(j-1)p + k},

    and the y-uncertainty becomes

    .. math::
        dy'_j = \frac{dy}{\sqrt{p}}.

    :param dataset: The dataset to be packed.
    :type dataset: experimentalis.data.Dataset

    :param packing_factor: The "packing factor" or block size
    :type packing_factor: int

    :returns: A new dataset of the packed data
    :rtype: experimentalis.data.Dataset
    """

    def pack_array(array: NDArray, block_size: int) -> NDArray:
        """
        Packs an individual dimension of the dataset.

        :param array: A numerical array.
        :type array: NDArray
        
        :param block_size: The block size for each packing.
        :type block_size: int

        :returns: A copy of the array after packing.
        """
        n_blocks = len(A) // block_size
        return A[:n_blocks * block_size]\
            .reshape(n_blocks, block_size)\
            .mean(axis=1)

    if not (dataset.dx == 0).all():
        warnings.warn("Packing a dataset with a nonzero dx is not recommended.")

    x = pack_array(dataset.x, packing_factor)
    y = pack_array(dataset.y, packing_factor)
    dx = np.zeros_like(x)
    dy = pack_array(dataset.dy, packing_factor) / np.sqrt(packing_factor)

    return Dataset(x=x, y=y, dx=dx, dy=dy)

def calculate_uncertainty(raw_data: Dataset,
                          method: str = "default",
                          indices_range: tuple[int] = None,
                          y_range: tuple[int] = None, 
                          plot: bool = False,
                          graphing_options: GraphingOptions = None):
    """
    Calculates the uncertainty :math:`dy` in some measurement series.
    Implicitly assumes that the uncertainty is already unknown
    (:math:`dy = \\vec{0}`), so if :math:`dy \\neq \\vec{0}`, then this will
    throw an error.
    """

    if not (raw_data.dy == 0).all():
        warnings.warn("Calculating the uncertainty in dataset with already known uncertainty is not recommended.")
    
    if indices_range is None:
        indices_range = (0, raw_data.x.size)
    
    x_trimmed, y_trimmed = \
        map(lambda a: a[indices_range[0]:indices_range[1]],
            (raw_data.x, raw_data.y))
    indices_trimmed = np.arange(0, len(x_trimmed))
    
    data_trimmed = (x_trimmed, y_trimmed, indices_trimmed)
    
    x, y = raw_data
    indices = np.arange(0, len(x))
    
    if plot:  
        plt.figure()
        plt.xlim(indices_range[0], indices_range[1])
        if y_range is not None:
            plt.ylim(y_range[0], y_range[1])
        graphing_options.set_labels(xlabel='Index')
        plt.scatter(indices, y, marker='.')
        plt.show()
        
        hist, bins = np.histogram(y_trimmed, bins=20)
        plt.bar(bins[:-1], hist, width = bins[1]-bins[0])
        plt.ylim(0, 1.2 * np.max(hist))
        plt.xlabel(f'Raw {graphing_options.y_label} Value ({graphing_options.y_units})')
        plt.ylabel('Number of Occurences')
        plt.show()
    
    match method:
        case "digital":
            digital = (np.max(y_trimmed) - np.min(y_trimmed)) / (2 * np.sqrt(3))
            print('Digital Uncertainty:', digital)
            return digital
        case "default":
            return isolate_noise_uncertainty(data_trimmed)

def isolate_noise_uncertainty(raw_data: tuple[NDArray]):
    """
    Calculates uncertainty due to noise in a tuple dataset, calculated as
    the standard deviation over a period where the data should be uniform.

    :returns: The standard deviation over a uniform distribution.
    :rtype: float
    """
    x, y, indices = raw_data
    
    y_ave = np.mean(y)
    y_std = np.std(y)
    
    print('Mean = ', y_ave,y_std)
    print('Standard Deviation (Noise Value) = ', y_std)
    
    return y_std
