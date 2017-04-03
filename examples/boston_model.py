"""
Have fun with the number of epochs!
"""
import numpy as np
from sklearn.datasets import load_boston
from miniflow.layers import Input, Linear, Sigmoid, Variable
from miniflow.topology import Model
from sklearn.model_selection import train_test_split

# Load data
data = load_boston()
X = data['data']
y = data['target']
print("Shapes X: {} y:{}".format(X.shape, y.shape))

# Normalize data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# Split between train/test sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
print("Shapes X_train: {} y_train:{} X_test:{} y_test:{}".format(X_train.shape, y_train.shape, X_test.shape, y_test.shape))

# Configuration
n_features = X_train.shape[1]
n_hidden = 100

# Layers Weights:
W1 = np.random.randn(n_features, n_hidden)
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, 1)
b2 = np.zeros(1)

# Neural network inputs (X and y):
Xi = Input(name="X_input")
yi = Input(name="y_input")

# Neural Network trainable parameter:
W1i, b1i = Variable(name="W1"), Variable(name="b1")
W2i, b2i = Variable(name="W2"), Variable(name="b2")

# Topology
Xi = Input()
x = Linear(W1i, b1i)(Xi)
x = Sigmoid()(x)
x = Linear(W2i, b2i)(x)

feed_dict = {
    W1i: W1,
    b1i: b1,
    W2i: W2,
    b2i: b2
}

model = Model(inputs=[Xi], outputs=[x])
model.compile(loss='mse')
model.train(X_train, y_train, feed_dict=feed_dict, epochs=1000, batch_size=32)
