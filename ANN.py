from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dense

#load data
import tensorflow as tf
import numpy as np
mnist = tf.keras.datasets.mnist 
(x_train, y_train), (x_test, y_test) = mnist.load_data() 
x_train = x_train.reshape([len(x_train),28*28]) 
x_train = np.asarray(x_train).astype(np.float32) / 255.0
x_test = x_test.reshape([len(x_test),28*28]) 
x_test = np.asarray(x_test).astype(np.float32) / 255.0

