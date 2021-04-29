import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from predictions import *
from arima_model import ArimaPrediction
from lstm_model import LstmPredictor
from smartless_model import DelayPredictor

# Parameters
first_index_to_predict = 299

# Data
aapl = yf.Ticker('AAPL')
history = aapl.history(period='24mo')
aapl_stock = history['Open'].tolist()
aapl_stock = np.divide(aapl_stock, 150.)

googl = yf.Ticker('GOOGL')
history = googl.history(period='24mo')
googl_stock = history['Open'].tolist()
googl_stock = np.divide(googl_stock, 2400.)

stocks = aapl_stock
stocks = [aapl_stock, googl_stock]
stocks = np.asarray(stocks).transpose()

# plt.figure()
# plt.plot(aapl_stock, 'o-')
# plt.plot(googl_stock, 'o-')
# plt.show()

# Models
arima = ArimaPrediction({'p': 4, 'q': 0, 'd': 1})
lstm = LstmPredictor({'nb_timesteps': 10, 'nb_features': 2, 'nb_outputs': 1, 'epochs': 100})
delay = DelayPredictor()

# Predictions
models = [arima, lstm, delay]
models = [lstm]
models_prediction = ModelsPrediction(stocks, last_index_for_learning=first_index_to_predict-1)
models_prediction.learn(models)
models_prediction.predict()
kpi = models_prediction.measure_kpi(stocks[first_index_to_predict:])

# Display data
plt.figure()
plt.plot(stocks[first_index_to_predict:], 'o-')
for i in range(len(models)):
    plt.plot(models_prediction.prediction[i])
plt.show()

print(kpi)

print('end')

