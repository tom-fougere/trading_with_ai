import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

from lstm_model import lstm


aapl = yf.Ticker('AAPL')
history = aapl.history(period='24mo')
stock = history['Open'].tolist()

# plt.figure()
# plt.plot(stock, 'o-')
# plt.show()

timestep = 3

# Build datasets
X_train = []
y_train = []
for i in range(timestep, len(stock) - 1):
    X_train.append(np.divide(stock[i - timestep:i], 150.))
    y_train.append(np.divide(stock[i + 1], 150.))
X_train, y_train = np.array(X_train), np.array(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

model = lstm(timestep, 1, 1, units=50)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32)

predicted = np.multiply(model.predict(X_train), 150.)

plt.figure()
plt.plot(stock[timestep:], 'o-')
plt.plot(predicted)
plt.show()


print('end')
