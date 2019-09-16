import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Removing the first line: it contains extraneous text instead of the column titles. This
# text prevents the data set from being parsed properly by the pandas library:
my_dataset = pd.read_csv(‘data/my_dataset.csv’, skiprows=1,
low_memory=False)

# Removing columns with text explanations that we won’t need, url columns and other
# unnecessary columns:
my_dataset = my_dataset.drop([‘url’],axis=1)

# Removing all columns with only one value, or have more than 50% missing values to
# work faster (if your data set is large enough that it will still be meaningful):
my_dataset = my_dataset.dropna(thresh=half_count,axis=1)

# If the target column is the last one, you can create the matrix of dependent
# variables by typing:

X = dataset.iloc[:, :-1].values

# To have a vector of independent variables with only the data from the last column, you
# can type
y = dataset.iloc[:, -1].values

# Machine learning uses only numeric values (float or int data type). However, data sets
# often contain the object data type than needs to be transformed into numeric. In most
# cases, categorical values are discrete and can be encoded as dummy variables, assigning
# a number for each category. The simplest way is to use One Hot Encoder, specifying the
# index of the column you want to work on:

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# In a nutshell, we put all of our features
# into the same scale so that none are dominated by another

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# As you don’t need to fit it to your test set, you can just apply
# transformation.
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

# Save to CSV
# To be sure that you still have the raw data, it is a good practice to store the final output of
# each section or stage of your workflow in a separate csv file. In this way, you’ll be able to
# make changes in your data processing flow without having to recalculate everything.
my_dataset.to_csv(“processed_data/cleaned_dataset.csv”,index=False)
