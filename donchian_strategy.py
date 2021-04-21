from Utils.utils import sliding_window
import numpy as np


def donchian(data, period):
    signals = sliding_window(data, period, functions=[min, max])
    signals.append(np.divide(np.add(signals[0], signals[1]), 2.))
    return signals


def donchian_strategy(raw, donchian_signals, buy_signal, sell_signal):

    strategy_signal = np.zeros(len(raw))

    for i_index in range(len(raw)):
        if raw[i_index] > donchian_signals[buy_signal][i_index]:
            strategy_signal[i_index] = 1
        elif raw[i_index] < donchian_signals[sell_signal][i_index]:
            strategy_signal[i_index] = -1

    return strategy_signal


