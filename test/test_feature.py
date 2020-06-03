from feature import *
import numpy.testing as nptesting


def test_line_length():
    data = [[-1, 1, -1, 1], [2, -2, 2, -2]]
    nptesting.assert_array_equal(line_length(data), [2, 4])


def test_line_length_2():
    data = [[-1, 1, -1, 1], [2, -2, 2, -2]]
    nptesting.assert_array_equal(line_length_2(data), [2, 4])
