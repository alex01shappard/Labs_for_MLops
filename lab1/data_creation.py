#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd

def create_data(n, anomaly=False):
    days = np.arange(1, n + 1)
    # Базовая модель: периодическая функция для температуры
    temp = 15 + 10 * np.sin(2 * np.pi * days / n)
    # Добавляем шум
    noise = np.random.normal(0, 2, n)
    temp += noise
    if anomaly:
        # Добавляем аномалии: в 10% точек резко меняем значение
        n_anomalies = max(1, int(0.1 * n))
        indices = np.random.choice(n, n_anomalies, replace=False)
        temp[indices] += np.random.choice([15, -15], size=n_anomalies)
    data = pd.DataFrame({'day': days, 'temp': temp})
    return data

def main():
    # Создаем папки, если их нет
    os.makedirs('train', exist_ok=True)
    os.makedirs('test', exist_ok=True)

    # Генерируем данные для обучения
    train_data1 = create_data(100, anomaly=True)
    train_data2 = create_data(100, anomaly=False)

    # Генерируем данные для теста
    test_data1 = create_data(40, anomaly=True)
    test_data2 = create_data(40, anomaly=False)

    # Сохраняем данные в CSV
    train_data1.to_csv(os.path.join('train', 'train_data1.csv'), index=False)
    train_data2.to_csv(os.path.join('train', 'train_data2.csv'), index=False)
    test_data1.to_csv(os.path.join('test', 'test_data1.csv'), index=False)
    test_data2.to_csv(os.path.join('test', 'test_data2.csv'), index=False)

    print("Этап 1 (data_creation) завершен.")

if __name__ == '__main__':
    main()
