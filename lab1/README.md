# Лабораторная №1. Простейший MLOps Конвейер

Этот проект демонстрирует пример простейшего конвейера для автоматизации работы с моделью машинного обучения. Конвейер состоит из нескольких этапов, каждый из которых реализован в отдельном Python-скрипте, а их последовательный запуск осуществляется с помощью bash-скрипта.

## Этапы Конвейера

1. **Создание данных (`data_creation.py`)**  
   - Генерация синтетических данных, моделирующих изменение дневной температуры с добавлением шума и аномалий.
   - Разделение данных на обучающие и тестовые выборки. Данные для обучения сохраняются в папке `train`, а для тестирования — в папке `test`.

2. **Предобработка данных (`data_preprocessing.py`)**  
   - Считывание CSV-файлов из папок `train` и `test`.
   - Масштабирование данных (например, с помощью `sklearn.preprocessing.StandardScaler`).
   - Сохранение обработанных данных с префиксом `scaled_` в тех же папках.

3. **Обучение модели (`model_preparation.py`)**  
   - Объединение обработанных обучающих данных.
   - Обучение модели (полиномиальной линейной регрессии) для предсказания масштабированной температуры.
   - Сохранение обученной модели в файл `model.pkl`.

4. **Тестирование модели (`model_testing.py`)**  
   - Объединение обработанных тестовых данных.
   - Загрузка сохраненной модели и оценка её качества (например, вычисление R² score).
   - Вывод результата в виде строки, содержащей оценку метрики.

5. **Запуск конвейера (`pipeline.sh`)**  
   - Bash-скрипт, который последовательно запускает все вышеуказанные этапы.
   - При необходимости устанавливает необходимые зависимости (numpy, pandas, scikit-learn).

## Структура Проекта

```
.
└── lab1
    ├── data_creation.py
    ├── data_preprocessing.py
    ├── model_preparation.py
    ├── model_testing.py
    └── pipeline.sh
```

## Требования

- **Python 3.x**
- **pip**

Необходимые Python-пакеты (будут установлены автоматически через `pipeline.sh`):
- numpy
- pandas
- scikit-learn

## Инструкция по Запуску

1. **Клонируйте репозиторий и перейдите в каталог проекта:**

   ```bash
   cd lab1
   ```

2. **Сделайте bash-скрипт исполняемым:**

   ```bash
   chmod +x pipeline.sh
   ```

3. **Запустите конвейер:**

   ```bash
   ./pipeline.sh
   ```

После выполнения всех этапов в терминале будет выведена строка с оценкой метрики, например:

```
R2 score на тестовой выборке: 0.95
```

## Примечания

- Каждый этап конвейера можно модифицировать или дополнять под конкретные задачи.
- В данном примере используются синтетические данные. Для реальных проектов можно адаптировать скрипт `data_creation.py` для скачивания или получения данных из других источников.
- При необходимости, зависимости можно добавить или изменить в файле `pipeline.sh`.



