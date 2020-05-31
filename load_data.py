# first method -
# first you need to download flatten csv data, attached.
# second you need to separate the labels from the data and normalize the data.


load_data_train = pd.read_csv('mnist_train.csv')
x_train = load_data_train.iloc[:,1:].values / 255.0
y_train = load_data_train.iloc[:,0].values
load_data_test = pd.read_csv('mnist_test.csv')
x_test = load_data_test.iloc[:,1:].values / 255.0
y_test = load_data_test.iloc[:,0].values



# second method -
# use TensorFlow tutorial library.
# we get the data in 28*28 matric shape, so we need to flatten the data, and normalize it.

import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape([len(x_train),28*28]) 
x_train = np.asarray(x_train).astype(np.float32) / 255.0
x_test = x_test.reshape([len(x_test),28*28]) 
x_test = np.asarray(x_test).astype(np.float32) / 255.0


