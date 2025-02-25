#!/bin/bash
set -e  # Прерывание выполнения при ошибке

# Устанавливаем необходимые пакеты (при условии, что используется pip)
pip install --upgrade pip
pip install numpy pandas scikit-learn

# Запуск этапов конвейера
echo "Запуск этапа 1: создание данных..."
python data_creation.py

echo "Запуск этапа 2: предобработка данных..."
python data_preprocessing.py

echo "Запуск этапа 3: обучение модели..."
python model_preparation.py

echo "Запуск этапа 4: тестирование модели..."
python model_testing.py
