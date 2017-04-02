import numpy as np
from miniflow.layers import Layer


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

        self.m = self.inbounds[0].value.shape[0]
        # Save the computed output for backward.
        self.diff = y - a
        self.value = np.mean(self.diff**2)

    def backward(self):
        """
        Calculates the gradient of the cost.
        """
        self.gradients[self.y_true] = (2 / self.m) * self.diff
        self.gradients[self.inbounds[0]] = (-2 / self.m) * self.diff
