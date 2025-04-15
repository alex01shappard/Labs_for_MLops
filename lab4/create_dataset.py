import pandas as pd
from catboost.datasets import titanic

train, test = titanic()
dataset = train[['Pclass', 'Sex', 'Age']]
dataset.to_csv('datasets/titanic.csv', index=False)