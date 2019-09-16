import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scikitlearn as sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

my_dataset = pd.read_csv(‘data/my_dataset.csv’, skiprows=1, low_memory=False)
my_dataset = my_dataset.drop([‘url’], axis=1)
my_dataset = my_dataset.dropna(thresh=half_count, axis=1)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

onehotencoder = OneHotEncoder(categorical_features=0: xyz)
X = onehotencoder.fit_transform(X).toarray()

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

my_dataset.to_csv(“processed_data/cleaned_dataset.csv”, index=False)
