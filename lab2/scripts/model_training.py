import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle
import os

# Загрузка данных
X_train = pd.read_csv('lab2/train/X_train.csv')
y_train = pd.read_csv('lab2/train/y_train.csv').values.ravel()

# Путь к модели из lab1
model_path = 'lab1/model.pkl'

T = 30 # используемый период (должен совпадать с data_processing.py)

# Загрузка существующей модели или создание новой
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # Переобучение модели на новых данных
    model.fit(X_train, y_train)
else:
    # Создание новой модели, если файла нет
    model = LinearRegression()
    model.fit(X_train, y_train)

# Сохраняем модель вместе с параметром T
pipeline = {'T': T, 'model': model}
with open("lab2/model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

