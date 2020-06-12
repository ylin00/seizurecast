from src.utils import *
from numpy.testing import assert_array_equal


def test_save():
    data = [[1], [2]]
    label = ['a', 'b']
    save(data, label)
    d, l = load()
    assert_array_equal(d, data)
    assert_array_equal(l, label)
