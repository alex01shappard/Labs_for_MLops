#!/usr/bin/env python3
import os
import glob
import pandas as pd
import pickle
from sklearn.metrics import r2_score

def main():
    # Считываем обработанные файлы из test
    test_files = glob.glob(os.path.join("test", "scaled_*.csv"))
    if not test_files:
        print("Нет обработанных тестовых данных.")
        return

    dataframes = [pd.read_csv(f) for f in test_files]
    test_data = pd.concat(dataframes, ignore_index=True)

    if 'day' not in test_data.columns or 'temp_scaled' not in test_data.columns:
        print("Отсутствуют необходимые столбцы в тестовых данных.")
        return

    X_test = test_data[['day']]
    y_test = test_data['temp_scaled']

    # Загружаем модель
    if not os.path.exists("model.pkl"):
        print("Файл модели не найден.")
        return

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    predictions = model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    print("R2 score на тестовой выборке:", r2)

if __name__ == '__main__':
    main()
