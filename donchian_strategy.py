from Utils.utils import sliding_window
import numpy as np


def donchian_channel(data, period):
    """
    Compute the donchian channels (Sliding window of Min, max and mean of max/min)
    :param data: list of numbers, digit
    :param period: period used for the donchian channel
    :return:
        - channels, donchian channels, list of lists , digit
            - channels[0]: min
            - channels[1]: max
            - channels[2]: median
    """
    channels = sliding_window(data, period, functions=[min, max])
    channels.append(np.divide(np.add(channels[0], channels[1]), 2.))
    return channels


def donchian_strategy(raw, donchian_channels, buy_signal, sell_signal):
    """
    Compute strategy to buy/sell
    When raw signal > donchian channel => buy
    When raw signal < donchian channel => sell
    :param raw: raw signal, list of digit
    :param donchian_channels: list of donchian channels
    :param buy_signal: selection of signal in donchian channel to order to buy
    :param sell_signal: selection of signal in donchian channel to order to sell
    :return:
        - strategy signal, list of orders (1 = buy, -1 = sell, 0 = nothing)
    """

    strategy_signal = np.zeros(len(raw))

    for i_index in range(len(raw)):
        if raw[i_index] > donchian_channels[buy_signal][i_index]:
            strategy_signal[i_index] = 1
        elif raw[i_index] < donchian_channels[sell_signal][i_index]:
            strategy_signal[i_index] = -1

    return strategy_signal


