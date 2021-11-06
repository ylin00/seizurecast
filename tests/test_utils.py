import numpy as np

from seizurecast.utils import *
from numpy.testing import assert_array_equal

from seizurecast.utils import dataset_3d_to_2d


def disabled_test_save():
    data = [[1], [2]]
    label = ['a', 'b']
    save(data, label)
    d, l = load()
    assert_array_equal(d, data)
    assert_array_equal(l, label)


def test_dataset2Xy():
    ds_pwd = np.array([
        [[f"ep{ep}ch{ch}fe{fea}" for fea in range(0, 3)] for ch in range(0, 3)] for ep in range(0, 3)
    ])
    labels = ['excellent', 'good', 'fair']
    X, y = dataset2Xy(ds_pwd, labels)
    assert_array_equal(X[0], [
        'ep0ch0fe0', 'ep0ch1fe0', 'ep0ch2fe0',
        'ep0ch0fe1', 'ep0ch1fe1', 'ep0ch2fe1',
        'ep0ch0fe2', 'ep0ch1fe2', 'ep0ch2fe2'])
    assert_array_equal(y, labels)


def test_ds3to2():
    input = np.array([
        [[1,2,3],
         [4,5,6],
         [7,8,9]],
        [[9,8,7],
         [6,5,4],
         [3,2,1]]
    ])
    np.testing.assert_array_equal(
        dataset_3d_to_2d(input),
        np.array(
         [
             [1,9],
             [2,8],
             [3,7],
             [4,6],
             [5,5],
             *zip(range(6, 10, 1), range(4, 0, -1))
         ]
        )
    )