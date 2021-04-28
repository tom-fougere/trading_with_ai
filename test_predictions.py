import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from predictions import *
from arima_model import ArimaPrediction
from lstm_model import LstmPredictor

# Parameters
first_index_to_predict = 299

# Data
aapl = yf.Ticker('AAPL')
history = aapl.history(period='24mo')
stock = history['Open'].tolist()
stock = np.divide(stock, 150.)

# Models
arima = ArimaPrediction({'p': 4, 'q': 0, 'd': 1})
lstm = LstmPredictor({'nb_timesteps': 10, 'nb_features': 1})

# Predictions
models = [arima, lstm]
models_prediction = ModelsPrediction(stock, last_index_for_learning=first_index_to_predict-1, expected_data=[])
models_prediction.learn(models)
models_prediction.predict()

# Display data
plt.figure()
plt.plot(stock[first_index_to_predict:], 'o-')
for i in range(len(models)):
    plt.plot(models_prediction.prediction[i])
plt.show()

print('end')

