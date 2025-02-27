#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

def main():
    # Считываем обработанные файлы из train (файлы с префиксом scaled_)
    train_files = glob.glob(os.path.join("train", "scaled_*.csv"))
    if not train_files:
        print("Нет обработанных обучающих данных.")
        return

    dataframes = [pd.read_csv(f) for f in train_files]
    train_data = pd.concat(dataframes, ignore_index=True)

    if 'day' not in train_data.columns or 'temp_scaled' not in train_data.columns:
        print("Отсутствуют необходимые столбцы в данных.")
        return

    T = 30  # используемый период (должен совпадать с data_creation.py)
    # Формируем признаки только на основе циклических преобразований
    X = pd.DataFrame({
        'sin_day': np.sin(2 * np.pi * train_data['day'] / T),
        'cos_day': np.cos(2 * np.pi * train_data['day'] / T)
    })
    y = train_data['temp_scaled']

    model = LinearRegression()
    model.fit(X, y)

    # Сохраняем модель вместе с параметром T
    pipeline = {'T': T, 'model': model}
    with open("model.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    print("Этап 3 (model_preparation) завершен: модель обучена и сохранена.")

if __name__ == '__main__':
    main()
