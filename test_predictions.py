import yfinance as yf
import matplotlib.pyplot as plt
import sklearn.preprocessing as sklpp

from predictions import *
from arima_model import ArimaPrediction
from lstm_model import LstmPredictor
from smartless_model import DelayPredictor
from preprocessing_stocks import download_stocks_data

# Parameters
first_index_to_predict = 299

# Data
stocks, scaler = download_stocks_data(['AAPL'], ['max'], '5y')

# Models
arima = ArimaPrediction({'p': 4, 'q': 0, 'd': 1})
lstm = LstmPredictor({'nb_timesteps': 3, 'nb_features': stocks.shape[1],
                      'nb_outputs': 1, 'epochs': 500, 'batch_size': 50, 'activation': 'linear'})
delay = DelayPredictor()

# Predictions
# models = [arima, lstm, delay]
models = [delay, lstm]
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

