from utils import *

import pytest


def test_sliding_window():
    nums = [2, 8, 7, 4, 0, 5, 8, 9]

    # One function
    result1 = sliding_window(nums, 1, [max])
    result2 = sliding_window(nums, 3, [max])

    assert result1[0] == nums
    assert result2[0] == [8, 8, 7, 5, 8, 9]

    # Several functions
    mean_func = lambda a: sum(a)/len(a)

    result1 = sliding_window(nums, 1, [max, mean_func, min])
    result2 = sliding_window(nums, 3, [max, mean_func, min])

    assert result1[0] == nums
    assert result2[0] == [8, 8, 7, 5, 8, 9]
    assert result1[1] == nums
    assert result2[1] == pytest.approx([5.667, 6.334, 3.667, 3, 4.334, 7.334], 0.01)
    assert result1[2] == nums
    assert result2[2] == [2, 4, 0, 0, 0, 5]
