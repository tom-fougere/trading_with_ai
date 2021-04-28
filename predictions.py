from abc import ABC, abstractmethod


class ModelPrediction(ABC):

    @abstractmethod
    def learn(self, past_data, parameters):
        pass

    @abstractmethod
    def predict(self):
        pass


class Predictions:
    def __init__(self, past_data, expected_data):
        """

        :param past_data: data to learn, list of stocks, float
        :param expected_data: data to predict, list of stocks, float
        """

        self.past = past_data
        self.expectation = expected_data
        self.models = []
        self.parameters = []
        self.prediction = []
        self.kpi = []

    def learn(self, models, parameters):
        """
        Make the data learn on the past data
        :param models: models to predict the data, list of classes, prediction_class
        :param parameters: parameters of the used model, list of dict
        """

        self.models = models
        self.parameters = parameters

        for model, param in zip(self.models, self.parameters):
            model.learn(self.past_data, param)

    def predict(self):
        """
        Predict the data for all models and save prediction
        """

        for model in self.models:
            prediction = model.predict()
            self.prediction.append(prediction)

    def measure_kpi(self):
        """
        Measure KPI on all prediction/expectation
        :return
            - kpi: kpi of all models, list of kpi
        """

        for pred, expect in zip(self.prediction, self.expectation):
            self.kpi.append(pred - expect)

        return self.kpi
