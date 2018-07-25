# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 19:18:55 2018

@author: Kartik Chavan
Contact: +1 (817) 513-6107
Email: chavankartik94@yahoo.in


"""

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('train_ZoGVYWq.csv')
dataset = dataset.fillna((dataset.mean()))
X = dataset.iloc[:, 1:12].values
y = dataset.iloc[:, 12].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_9 = LabelEncoder()
X[:, 9] = labelencoder_X_9.fit_transform(X[:, 9])
labelencoder_X_8 = LabelEncoder()
X[:, 8] = labelencoder_X_8.fit_transform(X[:, 8])
onehotencoder = OneHotEncoder(categorical_features = [8,9])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
#from sklearn.model_selection import train_test_split
#X_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X)#_train)
#X_test = sc.transform(x_test)


# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 10, input_dim = 15, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 20, kernel_initializer = 'uniform', activation = 'relu'))

#classifier.add(Dense(output_dim = 10, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['mae','accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y, batch_size = 5, nb_epoch = 25)

############################################################
############################# DELETE #######################
############################################################
# Predicting the Test set results
#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#cm
#############################################################
########################## DELETE ###########################
#############################################################

##############################################################
########### Making Prediction on Final Dataset ##############
#############################################################
test_dataset = pd.read_csv('test_66516Ee.csv')
test_dataset = test_dataset.fillna((test_dataset.mean()))
test_X = test_dataset.iloc[:, 1:12].values
test_id = test_dataset.iloc[:, 0].values

# Encoding categorical data
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
test_labelencoder_X_9 = LabelEncoder()
test_X[:, 9] = test_labelencoder_X_9.fit_transform(test_X[:, 9])
test_labelencoder_X_8 = LabelEncoder()
test_X[:, 8] = test_labelencoder_X_8.fit_transform(test_X[:, 8])
test_onehotencoder = OneHotEncoder(categorical_features = [8,9])
test_X = test_onehotencoder.fit_transform(test_X).toarray()
test_X = test_X[:, 1:]

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
test_sc = StandardScaler()
test_X_test = test_sc.fit_transform(test_X)


#### Predicting ####
y_pred = classifier.predict(test_X_test)



########################################################
############## Optimization ###########################
#######################################################
P = []
Premium = test_X[:,14]

X = []
Increase_prob = []
for i in range(len(y_pred)):
    P.append(y_pred[i][0])
    #Premium.append(x_test)
    X.append(1650)
    Increase_prob.append(0.5)
    
from scipy.optimize import minimize
def objective(gues):
    return -((P[i] + gues[1]/100*P[i])*Premium[i] - gues[0])
        
#print(objective(gues))

def constraint1(gues):
    return gues[1] - (20*(1-math.e**(-10*(1-math.e**(-gues[0]/400))/5)))

def constraint2(gues):
    return

con1 = {'type': 'eq', 'fun': constraint1}
cons = [con1]

bnds = ((0,2000),(0,100))


ans_lst = []
for i in range(len(P)):
    guess = [X[i], Increase_prob[i]]
    sol = minimize(objective, guess, bounds = bnds, constraints= cons)#, method = 'SLSQP')
    ans_lst.append((test_id[i],P[i],sol.x[0]))

labels = ['id','renewal', 'incentives']

df = pd.DataFrame.from_records(ans_lst, columns=labels)

df.to_csv("file_path.csv", index = False)

