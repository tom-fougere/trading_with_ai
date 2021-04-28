import yfinance as yf
import matplotlib.pyplot as plt

from predictions import *
from arima_model import ArimaPrediction

# Parameters
first_index_to_predict = 299

# Data
aapl = yf.Ticker('AAPL')
history = aapl.history(period='24mo')
stock = history['Open'].tolist()

# Models
arima = ArimaPrediction({'p': 4, 'q': 0, 'd': 1})

# Predictions
models_prediction = ModelsPrediction(stock, last_index_for_learning=first_index_to_predict-1, expected_data=[])
models_prediction.learn([arima])
models_prediction.predict()

# Display data
plt.figure()
plt.plot(stock[first_index_to_predict:], 'o-')
plt.plot(models_prediction.prediction[0])
plt.show()

