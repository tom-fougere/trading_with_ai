import yfinance as yf
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklpp

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
aapl_stock = np.reshape(aapl_stock, (len(aapl_stock), 1))
aapl_scaler = sklpp.MinMaxScaler()
aapl_stock = aapl_scaler.fit_transform(aapl_stock)

googl = yf.Ticker('GOOGL')
history = googl.history(period='24mo')
googl_stock = history['Open'].tolist()
googl_stock = np.asarray(googl_stock)
googl_stock = np.reshape(googl_stock, (len(googl_stock), 1))
googl_scaler = sklpp.MinMaxScaler()
googl_stock = googl_scaler.fit_transform(googl_stock)

stocks = aapl_stock
stocks = [aapl_stock, googl_stock]
stocks = np.reshape(stocks, (stocks[0].shape[0], len(stocks)))

# plt.figure()
# plt.plot(aapl_stock, 'o-')
# plt.plot(googl_stock, 'o-')
# plt.show()

# Models
arima = ArimaPrediction({'p': 4, 'q': 0, 'd': 1})
lstm = LstmPredictor({'nb_timesteps': 10, 'nb_features': stocks.shape[1],
                      'nb_outputs': 1, 'epochs': 100, 'activation': 'linear'})
delay = DelayPredictor()

# Predictions
# models = [arima, lstm, delay]
models = [lstm]
models_prediction = ModelsPrediction(stocks, last_index_for_learning=first_index_to_predict-1)
models_prediction.learn(models)
models_prediction.predict()

# Metrics
kpi = models_prediction.measure_kpi(stocks[first_index_to_predict:])
for metrics in kpi:
    print(metrics['mse'])

# Display data
plt.figure()
plt.plot(stocks[first_index_to_predict:], 'o-')
for i in range(len(models)):
    plt.plot(models_prediction.prediction[i])
plt.show()

print(kpi)

print('end')

