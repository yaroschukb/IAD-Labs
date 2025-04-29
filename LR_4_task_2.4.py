import json
import requests
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt
from sklearn import covariance, cluster
import yfinance as yf
import logging

# Вимикаємо логування для 'yfinance'
logging.getLogger('yfinance').setLevel(logging.CRITICAL)

# Вхідний файл із символічними позначеннями компаній
response = requests.get('https://raw.githubusercontent.com/PacktPublishing/Artificial-Intelligence-with-Python/refs/heads/master/Chapter%2004/code/company_symbol_mapping.json')
company_symbols_map = json.load(StringIO(response.text))

symbols, names = np.array(list(company_symbols_map.items())).T

# Завантаження архівних даних котирувань
start_date = "2022-07-03"
end_date = "2024-05-04"

quotes = []
valid_symbols = []
valid_names = []

for symbol, name in zip(symbols, names):
    try:
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        if not data.empty or len(data.columns) <= 4:
          quotes.append(data)
          valid_symbols.append(symbol)
          valid_names.append(name)
    except Exception as e:
        pass
        
symbols = np.array(valid_symbols)
names = np.array(valid_names)

# Тепер створюємо нормальні масиви
opening_quotes = np.array([q["Open"].values.ravel() for q in quotes])
closing_quotes = np.array([q["Close"].values.ravel() for q in quotes])

# Обчислення різниці між відкриттям і закриттям
quotes_diff = closing_quotes - opening_quotes

# Нормалізація даних
X = quotes_diff.T
X /= X.std(axis=0)

# Створення моделі графа
edge_model = covariance.GraphicalLassoCV()

# Навчання моделі
with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Створення моделі кластеризації на основі поширення подібності
_, labels = cluster.affinity_propagation(edge_model.covariance_)
num_labels = labels.max()

# Виведення результатів кластеризації
for i in range(num_labels + 1):
    print("Cluster", i + 1, "==>", ', '.join(names[labels == i]))