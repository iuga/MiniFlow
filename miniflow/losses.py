import numpy as np
from miniflow.layers import Layer

"""
Loss/Cost/Objective Functions
-----------------------------
Function that maps an event or values of one or more variables into a real number intuitively representing
some "cost" associated with the event.
Intuitively, the loss will be high if we're doing a poor job of clasifying the training data, and it will
be low if we're doing well.

In other words, the loss function measures how compatible a given set of parameters is with respect to the
ground truth labels in the training dataset. It's defined in such way that making good predictions on the
training data is equivalent to having a small loss.
"""


class MSE(Layer):
    def __init__(self, y_true):
        """
        The mean squared error cost function.
        Should be used as the last node for a network.
        """
        self.y_true = y_true
        Layer.__init__(self)

    def forward(self):
        """
        Calculates the mean squared error.
        """
        # NOTE: We reshape these to avoid possible matrix/vector broadcast
        # errors.
        #
        # For example, if we subtract an array of shape (3,) from an array of shape
        # (3,1) we get an array of shape(3,3) as the result when we want
        # an array of shape (3,1) instead.
        #
        # Making both arrays (3,1) insures the result is (3,1) and does
        # an elementwise subtraction as expected.
        y = self.y_true.value.reshape(-1, 1)
        a = self.inbounds[0].value.reshape(-1, 1)

        self.m = self.y_true.value.shape[0]
        # Save the computed output for backward.
        self.diff = y - a
        self.value = np.mean(self.diff**2)

    def backward(self):
        """
        Calculates the gradient of the cost.
        """
        self.gradients[self.y_true] = (2 / self.m) * self.diff
        self.gradients[self.inbounds[0]] = (-2 / self.m) * self.diff
