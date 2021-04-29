import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def download_stocks_data(stocks_name, indicators=['Open'], period='max'):
    """
    Get stocks
    :param stocks_name: Name of wanted stocks, list of strings
    :param indicators: Indicators to extract (available indicators: 'Open', 'High', 'Low', 'Close', 'Volume', 'max'), list of strings
    :param period: Data period to download ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
    :return:
        - stocks (nb samples, features (stock_name * indicators))
        - scaler, minmaxscaler
    """

    # Data
    data = []
    for stock_name in stocks_name:
        ticker = yf.Ticker(stock_name)
        history = ticker.history(period=period)

        if 'max' in indicators:
            indicators = ['Open', 'High', 'Low', 'Close', 'Volume']

        for indicator in indicators:
            single_column = history[indicator].tolist()
            data.append(single_column)

    stocks = np.asarray(data).transpose()
    scaler = MinMaxScaler()
    stocks = scaler.fit_transform(stocks)

    return stocks, scaler
