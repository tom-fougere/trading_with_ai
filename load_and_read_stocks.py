import yfinance as yf
import matplotlib.pyplot as plt


aapl = yf.Ticker('AAPL')
history = aapl.history(period='1mo')

history = history.drop(['Dividends', 'Stock Splits'], axis=1)
history['Middle'] = (history['Open'] + history['Close']) /2.

print(history.head(10))

x = history['Volume'][0]

plt.figure()
plt.subplot(211)
plt.plot(history['Middle'], 'o-')
plt.xticks(rotation=45)
plt.subplot(212)
plt.bar(history.index.values, history['Volume'])
plt.xticks(rotation=45)
plt.show()

print('end')
