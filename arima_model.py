from statsmodels.tsa.arima.model import ARIMA

from predictions import ModelPrediction


class ArimaPrediction(ModelPrediction):
    def __init__(self, parameters):
        super().__init__()

        # Number of lag observations, integer
        self.p = parameters['p']
        # Degree of differencing, integer
        self.d = parameters['d']
        # size/width of the moving average window, integer
        self.q = parameters['q']

    def learn(self, data=[], last_index_to_learn=[]):
        """
        No learning stage for arima prediction
        """
        pass

    def predict(self, data, first_index_to_predict):
        """
        Predict outputs with an arima forecast method
        :param data: all data, including past and future
        :param first_index_to_predict: First index to predict the output
        :return:
            - predictions, list of values
        """
        predictions = []
        num_prediction = len(data) - first_index_to_predict

        for i_data in range(num_prediction):
            history = data[:first_index_to_predict + i_data]
            model = ARIMA(history, order=(self.p, self.d, self.q))
            model_fit = model.fit()
            output = model_fit.forecast()
            predictions.append(output[0])

        return predictions
