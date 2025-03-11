import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob

def main():
    # Считываем новые данные из папки train
    train_files = glob.glob(os.path.join("lab1/train", "scaled_*.csv"))
    if not train_files:
        print("Нет новых данных для обработки.")
        return
    data_train=pd.concat([pd.read_csv(f) for f in train_files],ignore_index=True)
    
    # Считываем обработанные файлы из test (файлы с префиксом scaled_)
    test_files = glob.glob(os.path.join("lab1/test", "scaled_*.csv"))
    if not test_files:
        print("Нет обработанных тестовых данных.")
        return

    dataframes = [pd.read_csv(f) for f in test_files]
    data_test = pd.concat(dataframes, ignore_index=True)
    
    # Создание признаков
    T = 30 # фиксированный период
    X_train = pd.DataFrame({
        'sin_day': np.sin(2 * np.pi * data_train['day'] / T),
        'cos_day': np.cos(2 * np.pi * data_train['day'] / T)
    })
    y_train = data_train['temp_scaled']
    
    X_test = pd.DataFrame({
       'sin_day': np.sin(2 * np.pi * data_test['day'] / T),
        'cos_day': np.cos(2 * np.pi * data_test['day'] / T)
    })
    y_test = data_test['temp_scaled']
   
    # Сохранение обновленных датасетов
    X_train.to_csv(os.path.join('lab2/train','X_train.csv'), index=False)
    X_test.to_csv(os.path.join('lab2/test','X_test.csv'), index=False)
    y_train.to_csv(os.path.join('lab2/train','y_train.csv'), index=False)
    y_test.to_csv(os.path.join('lab2/test','y_test.csv'), index=False)

if __name__ == '__main__':
    main()