# CNN MNIST
# accuracy - 0.998
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist 
input = 28
#  reshape to have one channel to fit the cnn
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape(len(x_train), input, input, 1)  / 255.0
x_test = np.asarray(x_test).reshape(len(x_test), input, input, 1) / 255.0

y_train = np.asarray(y_train).reshape(len(y_train),1)
y_test = np.asarray(y_test).reshape(len(y_test),1)

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categories='auto')
y_train = onehotencoder.fit_transform(y_train).toarray()
y_test= onehotencoder.fit_transform(y_test).toarray()

# we have two convolutional layers before input to the ANN
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape = (28, 28, 1), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(28,(3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Flatten())
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# plot performance and calculate the accuracy
history = classifier.fit(x_train, y_train,  validation_data=(x_test, y_test), batch_size = 10, nb_epoch = 100)
loss, accuracy  = classifier.evaluate(x_test, y_test, verbose=False)
print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='best')
plt.show()


