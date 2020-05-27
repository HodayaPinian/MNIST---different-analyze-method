## SVM - support vector machine

import numpy as np
import pandas as pd

load_data_train = pd.read_csv('mnist_train.csv')
x_train = load_data_train.iloc[:,1:].values / 255.0 
y_train = load_data_train.iloc[:,0].values
load_data_test = pd.read_csv('mnist_test.csv')
x_test = load_data_test.iloc[:,1:].values / 255.0
y_test = load_data_test.iloc[:,0].values

#SVM classifer
from sklearn.svm import SVC
classifier = SVC(C=5, gamma=0.05)      
classifier.fit(x_train, y_train)

#cross validation accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)

# the best parameters are:
# for self check, you can use gridSearch- 
"""
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [13, 4, 5, 7, 10, 100], 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]},
              {'C': [13, 4, 5, 7, 10, 100], 'kernel': ['rbf'], 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]},
              {'C': [13, 4, 5, 7, 10, 100], 'kernel': ['poly'], 'gamma': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
"""
