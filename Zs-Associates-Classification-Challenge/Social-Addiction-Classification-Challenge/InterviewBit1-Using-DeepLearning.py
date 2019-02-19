
#Zs Data Science Competition Organised by InterviewBit 7th Dec 2018
#This was my first Attempt at any online Data-Science Competition, finished 61st in this competition!

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')


#Cleaning the dataset
#Cleaning data with pandas, scikitlearn imputer can also be used
dataset = dataset.fillna(dataset.mean())
dataset = dataset.drop(['country', 'social_account_number'], axis=1)
#Removing Country And Social_Account_Number Columns!


#Writting Custom function to convert yes or no into numbers!

#Creating Feature & Target Matrices
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values





# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 9]= labelencoder_X_1.fit_transform(X[:, 9])
labelencoder_X_2 = LabelEncoder()
X[:, 14]= labelencoder_X_2.fit_transform(X[:, 14])

'''
onehotencoder = OneHotEncoder(categorical_features = [1])
#To avoid Dummy Variable Trap we use oneHotEncoder!
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
'''

#Converting Yes&No into 0 & 1
Xt = pd.DataFrame(X)
Xt = Xt.astype('float64')
X = Xt.values


#Data-Cleaning Done!


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)

# Since Feature Scaling is MANDATORY IN deep Learning
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#KERAS

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense #Initialize the weights close to 0, but not 0


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu', input_dim = 19))
#Add method is used to add a layer

# Adding the second hidden layer
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set, no rule of thumb here this is where we experiment
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

from sklearn import metrics
#Accuracy of our model
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1-Score:",metrics.f1_score(y_test, y_pred))
