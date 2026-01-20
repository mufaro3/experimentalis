import numpy as np
from experimentalis.dataset import Dataset
from experimentalis.data import *

def test_dataset_creation():
    """
    Simple dataset creation test.
    """
    x  = np.array([0, 1, 2])
    dx = np.zeros(3)
    y  = np.array([10, 20, 30])
    dy = np.ones(3)

    d = Dataset(x=x, dx=dx, y=y, dy=dy)

    assert len(d.x) == 3
    assert np.all(d.dy == 1)

def test_shear_dataset():
    d = Dataset(
        x = np.array([1, 2, 3, 4]),
        y = np.array([1, 2, 3, 4])
    )

    sheared = shear_dataset(d, 2)

    assert len(sheared.x) == 3
    assert 

def test_sort_dataset()

def test_trim_dataset()

def test_pack_data():
    """
    Tests pack_data for output correctness (but not statistically). 
    """
    x = np.arange(1000)
    y = np.ones(1000)

    dataset = pack_data((x, y), uncertainty=1.0, p=100)

    assert len(dataset.x) == 10
    np.testing.assert_allclose(dataset.y, 1.0)
    np.testing.assert_allclose(dataset.dy, 1.0 / np.sqrt(100))


def test_calculate_uncertainty()

def test_isolate_noise_uncertainty()
