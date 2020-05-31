## KNN- k nearest neighbor 
# accuracy - 0.985

import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape([len(x_train),28*28]) 
x_train = np.asarray(x_train).astype(np.float32) / 255.0
x_test = x_test.reshape([len(x_test),28*28]) 
x_test = np.asarray(x_test).astype(np.float32) / 255.0

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
#KNN classifer
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

classifier.fit(x_train, y_train)

#cross validation accuracy
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 10)
best_accuracies = accuracies.mean()
std_accuracies = accuracies.std()

#the best parameters are:
#for self check, you can use gridSearch to see what parameters are preferred- 

"""
from sklearn.model_selection import GridSearchCV
parameters = [[{'n_neighbors' : [5 ,10,50 ,100] }]]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(x_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
"""
