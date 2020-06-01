from keras.models import Sequential
from keras.layers import Dense

#load data
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist 
input = 28*28
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape([len(x_train),input]) 
x_train = np.asarray(x_train).astype(np.float32) / 255.0
x_test = x_test.reshape([len(x_test),input]) 
x_test = np.asarray(x_test).astype(np.float32) / 255.0

classifier = Sequential()
classifier.add(Dense(output_dim = input/7, init = 'uniform', activation = 'relu', input_dim = input))
classifier.add(Dense(output_dim = input/4, init = 'uniform', activation = 'relu'))
classifier.add(Dense(output_dim = 10, init = 'uniform', activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)




