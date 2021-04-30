import talos
import numpy as np
import matplotlib.pyplot as plt

from preprocessing_stocks import download_stocks_data
from lstm_model import LstmPredictor


# Data
stocks, scaler = download_stocks_data(['AAPL'], ['Close'], '1mo')


def talos_lstm_model(data, y_train, x_val, y_val, parameters):
    # Build model
    lstm = LstmPredictor({'nb_timesteps': parameters['nb_timesteps'],
                          'nb_features': parameters['nb_features'],
                          'nb_outputs': parameters['nb_outputs'],
                          'nb_layers': parameters['nb_layers'],
                          'units': parameters['units'],
                          'dropout': parameters['dropout'],
                          'activation': parameters['activation'],
                          'optimizer': parameters['optimizer'],
                          'loss': parameters['loss'],
                          'epochs': parameters['epochs'],
                          'batch_size': parameters['batch_size']})

    # Fit data
    lstm.learn(data, parameters['last_index_to_learn'], evaluation=True)

    return lstm.fit_history, lstm.lstm


# then we can go ahead and set the parameter space
params = {'nb_timesteps': [10],
          'nb_features': [1],
          'nb_outputs': [1],
          'nb_layers': [4],
          'units': [50],
          'dropout': [0.2],
          'optimizer': ['adam'],
          'loss': ['mean_squared_error'],
          'activation': ['linear'],
          'epochs': [100],
          'batch_size': [32],
          'last_index_to_learn': [15]}

params1 = {'nb_timesteps': 10,
          'nb_features': 1,
          'nb_outputs': 1,
          'nb_layers': 4,
          'units': 50,
          'dropout': 0.2,
          'optimizer': 'adam',
          'loss': 'mean_squared_error',
          'activation': 'linear',
          'epochs': 100,
          'batch_size': 32,
          'last_index_to_learn': 15}

hist, mo = talos_lstm_model(stocks, stocks, [], [], params1)

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.draw()

# and run the experiment
t = talos.Scan(x=stocks,
               y=np.zeros(stocks.shape),
               model=talos_lstm_model,
               params=params,
               experiment_name='lstm_test',
               val_split=0.,
               reduction_interval=10,
               reduction_window=1,
               reduction_threshold=0.1,)

# plt.figure()
plt.plot(t.round_history[0]['loss'])
plt.plot(t.round_history[0]['val_loss'])
plt.show()

print('end')
