from predictions import ModelPrediction


class DelayPredictor(ModelPrediction):
    def __init__(self, parameters={}):
        super().__init__()
        self.delay = parameters['delay'] if 'delay' in parameters.keys() else 1

    def learn(self, data, last_index_to_learn, evaluation=False):
        pass

    def predict(self, data, first_index_to_predict):
        return data[first_index_to_predict - self.delay:-self.delay]