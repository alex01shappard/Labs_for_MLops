import pandas as pd
import pickle
from sklearn.metrics import r2_score

# Загрузка данных
X_test = pd.read_csv('lab2/test/X_test.csv')
y_test = pd.read_csv('lab2/test/y_test.csv')

# Загрузка модели и параметра T
with open('lab2/model.pkl', 'rb') as f:
    pipeline = pickle.load(f)
model = pipeline['model']
T = pipeline['T']

# Предсказание и оценка
predictions = model.predict(X_test)
r2 = r2_score(y_test, predictions)
print("R2 score на тестовой выборке:", r2)