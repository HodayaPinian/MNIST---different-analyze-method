## SVM - support vector machine
# accuracy - 0.983

import numpy as np
import pandas as pd

load_data_train = pd.read_csv('mnist_train.csv')
x_train = load_data_train.iloc[:,1:].values / 255.0 
y_train = load_data_train.iloc[:,0].values
load_data_test = pd.read_csv('mnist_test.csv')
x_test = load_data_test.iloc[:,1:].values / 255.0
y_test = load_data_test.iloc[:,0].values

#warning - because this big data, it takes lot of time for calculate. you can choose specific numbers to check the algorithm on.
#example to choose specific numbers
"""
idx = (y_train ==  1) | (y_train == 3) | (y_train == 8) | (y_train == 4)
x_train = x_train[idx]
y_train = y_train[idx]

idx2 = (y_test ==  1) | (y_test == 3) | (y_test == 8) | (y_test == 4)
x_test = x_test[idx2]
y_test = y_test[idx2]
"""
#SVM classifer
from sklearn.svm import SVC
classifier = SVC(C=5, gamma=0.05)      
classifier.fit(x_train, y_train)

# accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

#cross validation accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
best_accuracies = accuracies.mean()
std_accuracies = accuracies.std()

