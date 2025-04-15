import pandas as pd

dataset = pd.read_csv('datasets/titanic.csv')
dataset = pd.get_dummies(dataset, columns=['Sex'], drop_first=True)
dataset.to_csv('datasets/titanic.csv', index=False)