from src.utils import dataset_3d_to_2d
import numpy as np


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

