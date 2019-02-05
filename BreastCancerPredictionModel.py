# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 22:48:08 2019

@author: sunny
"""

# breast Cancer prediction Model

#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# importing dataset
dataset = pd.read_csv('BreastCancerData.csv')
X = dataset.iloc[:,1:31]
y = dataset.iloc[:,31]

#taking care of missing data

dataset.isnull().sum

# taking care of categorical data

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

# spitting data into training and testing sets

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)

# feature scaling

from sklearn.preprocessing import StandardScaler
Sc_X = StandardScaler()
X_train = Sc_X.fit_transform(X_train)
X_test = Sc_X.transform(X_test)

# applying RandomForestClassifier model

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#predicting values for test set

y_pred = classifier.predict(X_test)

#checking the accuracy
Accuracy = accuracy_score(y_test,y_pred)
Accuracy = Accuracy*100

#  example
st = '0.07678	20.29	14.34 135.1 1297	0.1003	0.1328	0.198	0.1043	0.1809	0.05883	0.7572	0.7813	5.438	94.44	0.01149	0.02461	0.05688	0.01885	0.01756	0.005115	22.54	16.67	152.2	1575	0.1374	0.205	0.4	0.1625	0.2364'
st = st.split()
st = [float(xx) for xx in st]

sst = np.array([st])
y_pred = classifier.predict(sst)









