from abc import ABC, abstractmethod


class ModelPrediction(ABC):

    def __init__(self):
        self.first_index_to_predict = 0

    @abstractmethod
    def learn(self, data, last_index_to_learn):
        pass

    @abstractmethod
    def predict(self, data, first_index_to_predict):
        pass


class ModelsPrediction:
    def __init__(self, data, last_index_for_learning, expected_data):
        """
        :param data: data to learn/predict, list of stocks, float
        :param last_index_for_learning: last index in the data to use for model training, integer
        :param expected_data: data to predict, list of stocks, float
        """

        self.data = data
        self.last_index_for_learning = last_index_for_learning
        self.expectation = expected_data
        self.models = []
        self.prediction = []
        self.kpi = []

    def learn(self, models):
        """
        Make the data learn on the data devoted for learning
        :param models: models to predict the data, list of classes, prediction_class
        """

        self.models = models

        for model in self.models:
            model.learn(self.data, self.last_index_for_learning)

    def predict(self):
        """
        Predict the data for all models and save prediction
        """

        for model in self.models:
            prediction = model.predict(self.data, self.last_index_for_learning + 1)
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
