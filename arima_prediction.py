import yfinance as yf
import matplotlib.pyplot as plt

from arima_model import *

aapl = yf.Ticker('AAPL')
history = aapl.history(period='24mo')
stock = history['Open'].tolist()

history_range = 300
#
my_arima_model = ArimaPrediction({'p': 4, 'q': 0, 'd': 1})
my_arima_model.learn([], [])
class_pred = my_arima_model.predict(stock, history_range)

plt.figure()
plt.plot(stock[history_range:], 'o-')
plt.plot(class_pred)
plt.show()


print('end')
