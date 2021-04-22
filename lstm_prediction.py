from tensorflow.keras.layers import Input, Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential


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
    for i_layer in range(nb_layers):
        tensor = LSTM(units=units)(tensor)
        tensor = Dropout(dropout)(tensor)

    return tensor


def lstm(num_timesteps, num_features, num_outputs=1, nb_layers=4, units=50, dropout=0.2, activation='linear'):
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
    model = Sequential(inputs=inputs, outputs=outputs)

    return model
