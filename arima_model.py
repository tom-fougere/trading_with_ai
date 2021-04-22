from statsmodels.tsa.arima.model import ARIMA


def arima_forecast(nums, history_range, p=4, d=1, q=0):
    """
    Forecast values with arima method
    :param nums: values
    :param history_range: number of values for the history
    :param p: Number of lag observations, integer
    :param d: Degree of differencing, integer
    :param q: size/width of the moving average window, integer
    :return:
        - prediction: list of floats
    """
    predictions = []
    num_prediction = len(nums) - history_range

    for i_data in range(num_prediction):
        history = nums[i_data:history_range + i_data]
        model = ARIMA(history, order=(p, d, q))
        model_fit = model.fit()
        output = model_fit.forecast()
        predictions.append(output[0])

    return predictions
