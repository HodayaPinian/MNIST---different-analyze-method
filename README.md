# MNIST---different-analyze-method
Exeplore MNIST data with different ML and DL method - SVM , KNN , ANN , CNN

MNIST database provide us a large database of handwritten digits, which is a great start in the ML field.
It has a training set of 60,000 examples, and a test set of 10,000 examples.
Every image has 28*28 pixel, that provide us a scan of a handwritten digit from 0 to 9.
Each pixel symbolic the color by the numbers 0 - 255 on the grayscale.

For using the data on SVM ,KNN, ANN method we will flatten the image, so we will get 28*28 = 784 inputs.
Because the data is relative simplicity, we can use them for ML modlinig for classification.

You can get the data from mnist library( https://pypi.org/project/python-mnist/ ) or from TensorFlow tutorial library(https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist/load_data). 
Also i'm adding here the flatten (784 inputs) csv files of the training and test set with their label in the first row. (got them from excellent tutorial- https://www.python-course.eu/neural_network_mnist.php )

I upload two examples to how load the data, attached in load_data.py file.
