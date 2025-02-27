#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score

def main():
    # Считываем обработанные файлы из test (файлы с префиксом scaled_)
    test_files = glob.glob(os.path.join("test", "scaled_*.csv"))
    if not test_files:
        print("Нет обработанных тестовых данных.")
        return

    dataframes = [pd.read_csv(f) for f in test_files]
    test_data = pd.concat(dataframes, ignore_index=True)

    if 'day' not in test_data.columns or 'temp_scaled' not in test_data.columns:
        print("Отсутствуют необходимые столбцы в тестовых данных.")
        return

    # Загружаем модель и параметр T
    if not os.path.exists("model.pkl"):
        print("Файл модели не найден.")
        return

    with open("model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    T = pipeline['T']
    model = pipeline['model']

    # Преобразуем тестовые данные: только sin и cos от day
    X_test = pd.DataFrame({
        'sin_day': np.sin(2 * np.pi * test_data['day'] / T),
        'cos_day': np.cos(2 * np.pi * test_data['day'] / T)
    })
    y_test = test_data['temp_scaled']

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print("R2 score на тестовой выборке:", r2)

if __name__ == '__main__':
    main()
