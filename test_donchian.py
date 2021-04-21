from donchian_strategy import *

import pytest
import yfinance as yf
import matplotlib.pyplot as plt


aapl = yf.Ticker('AAPL')
history = aapl.history(period='12mo')

PLOT = False


def test_donchian():

    nums = [2, 8, 7, 4, 0, 5, 8, 9]
    period = 3

    donchian_signals = donchian(nums, period=period)

    assert donchian_signals[0] == [2, 4, 0, 0, 0, 5]
    assert donchian_signals[1] == [8, 8, 7, 5, 8, 9]
    assert donchian_signals[2] == pytest.approx([5, 6, 3.5, 2.5, 4, 7])


def test_donchian_strategy():

    nums = [2, 8, 7, 10, 0, 5, 8, 9]

    period = 3
    donchian_signals = donchian(nums, period=period)

    stock_period = nums[period:]
    result = donchian_strategy(stock_period, donchian_signals, 1, 0)
    assert result == pytest.approx([1, -1,  0, 0, 1])

    if PLOT:
        plt.figure()
        plt.subplot(211)
        plt.plot(stock_period, 'o-')
        plt.plot(donchian_signals[0])
        plt.plot(donchian_signals[1])
        plt.plot(donchian_signals[2])
        plt.subplot(212)
        plt.plot(result, 'o-')
        plt.show()
