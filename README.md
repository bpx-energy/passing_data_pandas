# passing_data_pandas
Gathering and preparing data for ML scoring & training. 


Data cleansing or data cleaning is the process of detecting and correcting (or removing)
corrupt or inaccurate records from a record set, table, or database and refers to identifying incomplete, incorrect, inaccurate or irrelevant parts of the data and then replacing,
modifying, or deleting the dirty or coarse data. //Wikipedia

Importing libraries
The absolutely first thing you need to do is to import libraries for data preprocessing.
There are lots of libraries available, but the most popular and important Python libraries
for working on data are Numpy, Matplotlib, and Pandas. Numpy is the library used for
all mathematical things. Pandas is the best tool available for importing and managing
datasets. Matplotlib (Matplotlib.pyplot) is the library to make charts.

# To make it easier for future use, you can import these libraries with a shortcut alias:
> import numpy as np
> import matplotlib.pyplot as plt
> import pandas as pd

Once you downloaded your data set and named it as a .csv file, you need to load it into a
pandas DataFrame to explore it and perform some basic cleaning tasks removing
information you don’t need that will make data processing slower.
Usually, such tasks include:
Removing the first line: it contains extraneous text instead of the column titles. This
text prevents the data set from being parsed properly by the pandas library:
my_dataset = pd.read_csv(‘data/my_dataset.csv’, skiprows=1,
low_memory=False)
Removing columns with text explanations that we won’t need, url columns and other
unnecessary columns:
my_dataset = my_dataset.drop([‘url’],axis=1)
Removing all columns with only one value, or have more than 50% missing values to
work faster (if your data set is large enough that it will still be meaningful):
my_dataset = my_dataset.dropna(thresh=half_count,axis=1)
It’s also a good practice to name the filtered data set differently to keep it separate from
the raw data. This makes sure you still have the original data in case you need to go back
to it.

Understanding the data
Now you have got your data set up, but you still should spend some time exploring it and
understanding what feature each column represents. Such manual review of the data set
is important, to avoid mistakes in the data analysis and the modelling process.
To make the process easier, you can create a DataFrame with the names of the columns,
data types, the first row’s values, and description from the data dictionary.
As you explore the features, you can pay attention to any column that:
is formatted poorly,
requires more data or a lot of pre-processing to turn into useful a feature, or
contains redundant information,
since these things can hurt your analysis if handled incorrectly.
You should also pay attention to data leakage, which can cause the model to overfit.
This is because the model will be also learning from features that won’t be available
when we’re using it to make predictions. We need to be sure our model is trained using
only the data it would have at the point of a loan application.
Deciding on a target column
With a filtered data set explored, you need to create a matrix of dependent variables and
a vector of independent variables. At first you should decide on the appropriate column
to use as a target column for modelling based on the question you want to answer. 
For example, if you want to predict the development of cancer, or the chance the credit will
be approved, you need to find a column with the status of the disease or loan granting ad
use it as the target column.
For example, if the target column is the last one, you can create the matrix of dependent
variables by typing:
X = dataset.iloc[:, :-1].values
That first colon (:) means that we want to take all the lines in our dataset. : -1 means
that we want to take all of the columns of data except the last one. The .values on the
end means that we want all of the values.
To have a vector of independent variables with only the data from the last column, you
can type
y = dataset.iloc[:, -1].values

Finally, it’s time to do the preparatory work to feed the features for ML algorithms. To
clean the data set, you need to handle missing values and categorical features,
because the mathematics underlying most machine learning models assumes that the
data is numerical and contains no missing values. Moreover, the scikit-learn library
returns an error if you try to train a model like linear regression and logistic regression
using data that contain missing or non-numeric values.
Dealing with Missing Values
Missing data is perhaps the most common trait of unclean data. These values usually
take the form of NaN or None.
here are several causes of missing values: sometimes values are missing because they do
not exist, or because of improper collection of data or poor data entry. For example, if
someone is under age, and the question applies to people over 18, then the question will
contain a missing value. In such cases, it would be wrong to fill in a value for that
question.
There are several ways to fill up missing values:
you can remove the lines with the data if you have your data set is big enough and
the percentage of missing values is high (over 50%, for example);
you can fill all null variables with 0 is dealing with numerical values;
you can use the Imputer class from the scikit-learn library to fill in missing values
with the data’s (mean, median, most_frequent)
you can also decide to fill up missing values with whatever value comes directly after
it in the same column.
These decisions depend on the type of data, what you want to do with the data, and the
cause of values missing. In reality, just because something is popular doesn’t necessarily
make it the right choice. The most common strategy is to use the mean value, but
depending on your data you may come up with a totally different approach.


Machine learning uses only numeric values (float or int data type). However, data sets
often contain the object data type than needs to be transformed into numeric. In most
cases, categorical values are discrete and can be encoded as dummy variables, assigning
a number for each category. The simplest way is to use One Hot Encoder, specifying the
index of the column you want to work on:
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
Dealing with inconsistent data entry
Inconsistency occurs, for example, when there are different unique values in a column
which are meant to be the same. You can think of different approaches to capitalization,
simple misprints and inconsistent formats to form an idea. One of the ways to remove
data inconsistencies is by to remove whitespaces before or after entry names and by
converting all cases to lower cases.
If there is a large number of inconsistent unique entries, however, it is impossible to
manually check for the closest matches. You can use the Fuzzy Wuzzy package to
identify which strings are most likely to be the same. It takes in two strings and returns a
ratio. The closer the ratio is to 100, the more likely you will unify the strings.
Handling Dates and Times
A specific type of data inconsistency is inconsistent format of dates, such as dd/mm/yy
and mm/dd/yy in the same columns. Your date values might not be in the right data
type, and this will not allow you effectively perform manipulations and get insight from
it. This time you can use the datetime package to fix the type of the date.
Scaling and Normalization
Scaling is important if you need to specify that a change in one quantity is not equal to
another change in another. With the help of scaling you ensure that just because some
features are big they won’t be used as a main predictor. For example, if you use the age
and the salary of a person in prediction, some algorithms will pay attention to the salary
more because it is bigger, which does not make any sense.

Normalization involves transforming or converting your dataset into a normal
distribution. Some algorithms like SVM converge far faster on normalized data, so it
makes sense to normalize your data to get better results.
There are many ways to perform feature scaling. In a nutshell, we put all of our features
into the same scale so that none are dominated by another. For example, you can use the
StandardScaler class from the sklearn.preprocessing package to fit and transform your
data set:
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
As you don’t need to fit it to your test set, you can just apply
transformation.
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)
Save to CSV
To be sure that you still have the raw data, it is a good practice to store the final output of
each section or stage of your workflow in a separate csv file. In this way, you’ll be able to
make changes in your data processing flow without having to recalculate everything.
As we did previously, you can store your DataFrame as a .csv using the pandas to_csv()
function.
my_dataset.to_csv(“processed_data/cleaned_dataset.csv”,index=False)
