import pandas as pd

dataset = pd.read_csv('datasets/titanic.csv')
dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset.to_csv('datasets/titanic.csv', index=False)