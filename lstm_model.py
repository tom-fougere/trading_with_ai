from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Model
import numpy as np

from predictions import ModelPrediction


def lstm_block(input_tensor, nb_layers, units, dropout):
    """
    Add lstm layers with the parameters passed to it
    :param input_tensor: the input tensor
    :param nb_layers: number of layers
    :param units: numbers of lstm neurons
    :param dropout: fraction of the input units to drop
    :return:
        - tensor: tensor of output features
    """

    tensor = input_tensor
    for i_layer in range(nb_layers-1):
        tensor = LSTM(units=units, return_sequences=True)(tensor)
        tensor = Dropout(dropout)(tensor)

    tensor = LSTM(units=units)(tensor)
    tensor = Dropout(dropout)(tensor)

    return tensor


def lstm(num_timesteps, num_features, num_outputs=1, nb_layers=4, units=50, dropout=0.2, activation=None):
    """
    Build lstm model
    :param num_timesteps: number of timesteps in the input
    :param num_features: number of features in the input
    :param num_outputs: number of outputs
    :param nb_layers: number of layers
    :param units: numbers of lstm neurons
    :param dropout: fraction of the input units to drop
    :param activation: activation function of the last (Dense) layer
    :return:
        - model: lstm model
    """

    # Specify the input shape
    inputs = Input(shape=(num_timesteps, num_features))

    # Build the layers
    features = lstm_block(inputs, nb_layers=nb_layers, units=units, dropout=dropout)

    # Specify the output shape
    outputs = Dense(units=num_outputs, activation=activation)(features)

    # Create the model with defined inputs and outputs
    model = Model(inputs=inputs, outputs=outputs)

    return model


class LstmPredictor(ModelPrediction):
    def __init__(self, parameters):
        super().__init__()

        # Extract model parameters
        self.nb_timesteps = parameters['nb_timesteps']
        self.nb_features = parameters['nb_features']
        self.nb_outputs = parameters['nb_outputs'] if 'nb_outputs' in parameters.keys() else 1
        self.day_prediction = parameters['day_prediction'] if 'day_prediction' in parameters.keys() else 1
        nb_layers = parameters['nb_layers'] if 'nb_layers' in parameters.keys() else 4
        units = parameters['units'] if 'units' in parameters.keys() else 50
        dropout = parameters['dropout'] if 'dropout' in parameters.keys() else 0.2
        activation = parameters['activation'] if 'activation' in parameters.keys() else None

        # build model
        self.lstm = lstm(self.nb_timesteps, self.nb_features, self.nb_outputs, nb_layers, units, dropout, activation)
        self.fit_history = []

        # Extract compilation parameters
        self.optimizer = parameters['optimizer'] if 'optimizer' in parameters.keys() else 'adam'
        self.loss = parameters['loss'] if 'loss' in parameters.keys() else 'mean_squared_error'
        self.epochs = parameters['epochs'] if 'epochs' in parameters.keys() else 100
        self.batch_size = parameters['batch_size'] if 'batch_size' in parameters.keys() else 32

    def learn(self, data, last_index_to_learn, evaluation):

        # Build datasets
        x_train = []
        y_train = []
        for i in range(self.nb_timesteps, last_index_to_learn):
            x_train.append(data[i - self.nb_timesteps:i])
            y_train.append(data[i, :self.nb_outputs])

        # Convert to tensor (samples, timesteps, features)
        x_train, y_train = np.array(x_train), np.array(y_train)

        if evaluation is True:
            x_test = []
            y_test = []
            for i in range(last_index_to_learn + 1, data.shape[0]):
                x_test.append(data[i - self.nb_timesteps:i])
                y_test.append(data[i, :self.nb_outputs])

            # Convert to tensor (samples, timesteps, features)
            x_test, y_test = np.array(x_test), np.array(y_test)
            validation_data = (x_test, y_test)
        else:
            validation_data = None

        self.lstm.compile(optimizer=self.optimizer, loss=self.loss)
        self.fit_history = self.lstm.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                                         validation_data=validation_data)

    def predict(self, data, first_index_to_predict):
        # Build dataset
        x_test = []
        for i in range(first_index_to_predict, data.shape[0]):
            x_test.append(data[i - self.nb_timesteps:i])

        # Convert to tensor (samples, timesteps, features)
        x_test = np.array(x_test)

        prediction = self.lstm.predict(x_test)

        return prediction
