import numpy as np
from sklearn.datasets import load_boston
from miniflow.layers import Input, Linear, Sigmoid, Variable
from miniflow.topology import Model
from sklearn.model_selection import train_test_split


def test_e2e_training_with_boston_dataset():
    """
    Train a full network with the boston dataset and check the loss.
    This small network must converge.
    """
    #
    # Pre-processing
    #
    # Load the dataset
    data = load_boston()
    X, y = data['data'], data['target']
    # Normalize the data
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    # Split between train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #
    # Define Network Topology
    #
    # Layers Initialization Weights:
    W1 = np.random.normal(0.1, 0.01, (13, 10))
    b1 = np.zeros(10)
    W2 = np.random.normal(0.1, 0.01, (10, 1))
    b2 = np.zeros(1)
    # Neural network inputs (X and y):
    Xi = Input(name="X_input")
    # Neural Network trainable parameter:
    W1i, b1i = Variable(name="W1"), Variable(name="b1")
    W2i, b2i = Variable(name="W2"), Variable(name="b2")
    # Topology
    Xi = Input()
    x = Linear(W1i, b1i)(Xi)
    x = Sigmoid()(x)
    x = Linear(W2i, b2i)(x)

    #
    # Training
    #
    # Define the base Model object
    model = Model(inputs=[Xi], outputs=[x])
    # Compile the model setting the loss funciton: Mean Square Error
    model.compile(loss='mse')
    # Train the model getting the history data:
    history = model.train(X_train, y_train, X_test=X_test, y_test=y_test, epochs=70, batch_size=32, feed_dict={
        W1i: W1,
        b1i: b1,
        W2i: W2,
        b2i: b2
    })

    # Assertions:
    # First Iteration the loss is over 200
    # In the last iteration the loss should be less than 20
    assert history['train_loss'][0] > 100
    assert history['train_loss'][-1] < 25
    assert history['test_loss'][0] > 100
    assert history['test_loss'][-1] < 25
