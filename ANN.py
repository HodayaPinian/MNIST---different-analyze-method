# ANN
# accuracy -  0.986
from keras.models import Sequential
from keras.layers import Dense

#load data
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist 
input = 28*28
# Dense needs int for input
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape([len(x_train),input]) 
x_test = x_test.reshape([len(x_test),input]) 
y_train = np.asarray(y_train).reshape(len(y_train),1)
y_test = np.asarray(y_test).reshape(len(y_test),1)

#for output layer, we need to encode the labels to 0/1
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories='auto')
y_train = onehotencoder.fit_transform(y_train).toarray()
y_test= onehotencoder.fit_transform(y_test).toarray()

#ANN network
classifier = Sequential()
classifier.add(Dense(112, activation="relu", input_dim=784, kernel_initializer="uniform"))
classifier.add(Dense(28, activation = 'relu', kernel_initializer = 'uniform'))
classifier.add(Dense( 10, activation = 'sigmoid',  kernel_initializer = 'uniform'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train, y_train, batch_size = 10, epoch = 100)

loss, accuracy  = classifier.evaluate(x_test, y_test, verbose=False)

#for visualize the accuracy through the epochs: 
"""
import matplotlib.pyplot as plt
history =classifier.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size = 10, nb_epoch = 100)
print(history.history.keys()) #names
plt.plot(history.history(['accuracy']))
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()

"""



