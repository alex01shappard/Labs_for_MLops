#!/usr/bin/env python3
import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_folder(folder, scaler=None, fit_scaler=False):
    files = glob.glob(os.path.join(folder, "*.csv"))
    dataframes = []
    for file in files:
        df = pd.read_csv(file)
        if 'temp' in df.columns:
            dataframes.append(df)
    if not dataframes:
        return None, files

    combined = pd.concat(dataframes, ignore_index=True)
    if fit_scaler:
        scaler = StandardScaler()
        scaler.fit(combined[['temp']])
    return scaler, files

def main():
    # Обучаем StandardScaler на объединенных данных из train
    scaler, train_files = process_folder("train", fit_scaler=True)
    if scaler is None:
        print("Нет данных для обучения.")
        return

    # Применяем scaler к файлам в train
    for file in train_files:
        df = pd.read_csv(file)
        if 'temp' in df.columns:
            df['temp_scaled'] = scaler.transform(df[['temp']])
            new_file = os.path.join("train", "scaled_" + os.path.basename(file))
            df.to_csv(new_file, index=False)
    print("Предобработка обучающих данных завершена.")

    # Применяем scaler к файлам в test
    _, test_files = process_folder("test", scaler=scaler, fit_scaler=False)
    for file in test_files:
        df = pd.read_csv(file)
        if 'temp' in df.columns:
            df['temp_scaled'] = scaler.transform(df[['temp']])
            new_file = os.path.join("test", "scaled_" + os.path.basename(file))
            df.to_csv(new_file, index=False)
    print("Предобработка тестовых данных завершена.")

if __name__ == '__main__':
    main()
