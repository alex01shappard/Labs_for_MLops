#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd


def create_data(n, anomaly=False):
    days = np.arange(1, n + 1)
    T = 30  # фиксированный период
    # Основной сигнал: синусоида с амплитудой 10 и базовым значением 15
    temp = 15 + 10 * np.sin(2 * np.pi * days / T)
    # Добавляем менее выраженный шум
    noise = np.random.normal(0, 0.5, n)
    temp += noise
    if anomaly:
        n_anomalies = max(1, int(0.1 * n))
        indices = np.random.choice(n, n_anomalies, replace=False)
        # Уменьшаем амплитуду аномалий
        temp[indices] += np.random.choice([2, -2], size=n_anomalies)
    data = pd.DataFrame({"day": days, "temp": temp})
    return data


def main():
    os.makedirs("train", exist_ok=True)
    os.makedirs("test", exist_ok=True)

    # Для тренировочных данных: один набор без аномалий и один с незначительными аномалиями
    train_data1 = create_data(100, anomaly=False)
    train_data2 = create_data(100, anomaly=True)

    # Для тестовых данных аналогичным образом
    test_data1 = create_data(40, anomaly=False)
    test_data2 = create_data(40, anomaly=True)

    train_data1.to_csv(os.path.join("train", "train_data1.csv"), index=False)
    train_data2.to_csv(os.path.join("train", "train_data2.csv"), index=False)
    test_data1.to_csv(os.path.join("test", "test_data1.csv"), index=False)
    test_data2.to_csv(os.path.join("test", "test_data2.csv"), index=False)

    print("Этап 1 (data_creation) завершен.")


if __name__ == "__main__":
    main()
