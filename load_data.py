# first methood -
# first you need to donwload fllerten csv data, attached.
#second yoy need to sperat the labels from the data and normlaiz the data.


load_data_train = pd.read_csv('mnist_train.csv')
x_train = load_data_train.iloc[:,1:].values / 255.0
y_train = load_data_train.iloc[:,0].values
load_data_test = pd.read_csv('mnist_test.csv')
x_test = load_data_test.iloc[:,1:].values / 255.0
y_test = load_data_test.iloc[:,0].values

