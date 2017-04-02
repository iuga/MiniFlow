"""
Have fun with the number of epochs!
"""
import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample
from miniflow.layers import Input, Linear, Sigmoid
from miniflow.losses import MSE
from miniflow.engine import topological_sort, forward_and_backward, sgd_update
from miniflow.topology import Model

# Load data
data = load_boston()
X = data['data']
y = data['target']
print("Shapes X: {} y:{}".format(X.shape, y.shape))

# Normalize data
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
n_features = X.shape[1]
n_hidden = 10

# Layers Weights:
W1 = np.random.randn(n_features, n_hidden)
b1 = np.zeros(n_hidden)
W2 = np.random.randn(n_hidden, 1)
b2 = np.zeros(1)

# Neural network:
Xi, yi = Input(), Input()
W1i, b1i = Input(), Input()
W2i, b2i = Input(), Input()

# Topology
Xi = Input()
x = Linear(W1i, b1i)(Xi)
x = Sigmoid()(x)
x = Linear(W2i, b2i)(x)
cost = MSE(yi)(x)

model = Model(inputs=[Xi], outputs=[cost])
model.train()
model.summary()

feed_dict = {
    Xi: X,
    yi: y,
    W1i: W1,
    b1i: b1,
    W2i: W2,
    b2i: b2
}

epochs = 2000
# Total number of examples
m = X.shape[0]
batch_size = 64
steps_per_epoch = m // batch_size

graph = topological_sort(feed_dict)
trainables = [W1i, b1i, W2i, b2i]

print("Total number of examples = {}".format(m))

# Step 4
for i in range(epochs):
    loss = 0
    for j in range(steps_per_epoch):
        # Step 1
        # Randomly sample a batch of examples
        X_batch, y_batch = resample(X, y, n_samples=batch_size)

        # Reset value of X and y Inputs
        Xi.value = X_batch
        yi.value = y_batch

        # Step 2
        forward_and_backward(graph)

        # Step 3
        sgd_update(trainables, learning_rate=0.05)

        loss += graph[-1].value

    print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))
