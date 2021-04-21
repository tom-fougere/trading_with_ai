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


def lstm(nb_layers, units=50, dropout=0.2):

    # specify the input shape
    inputs = Input(shape=(None,))

    features = lstm_block(inputs, nb_layers=nb_layers, units=units, dropout=dropout)
    outputs = Dense(units=1)(features)

    model = Sequential(inputs=inputs, outputs=outputs)
