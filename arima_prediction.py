import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from arima_model import *

aapl = yf.Ticker('AAPL')
history = aapl.history(period='24mo')
stock = history['Open'].tolist()

history_range = 300
prediction = arima_forecast(stock, history_range)

plt.figure()
plt.plot(stock[history_range:], 'o-')
plt.plot(prediction)
plt.show()


print('end')
