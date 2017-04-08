import numpy as np
import numpy.testing as npt
from miniflow.layers import Softmax, Input, ReLU, Sigmoid, Identity
from miniflow.losses import MSE


def test_softmax_layer():
    """
    The softmax function takes an N-dimensional vector of arbitrary real values and
    produces another N-dimensional vector with real values in the range (0, 1) that add up to 1.0.
    It's easy to see that the results are always positive (because of the exponents).
    Moreover, since the numerator appears in the denominator summed up with some
    other positive numbers. Therefore, it's in the range (0, 1).

    Softmax function is used to "collapse" the logits into a vector of probabilities
    denoting the probability of x belonging to each one of the T output classes

    Links: https://gist.github.com/stober/1946926
    """
    x = Input()
    layer = Softmax()(x)

    # Scenario 1:
    x.value = [0.1, 0.2]
    layer.forward()
    npt.assert_almost_equal(layer.value, [0.47502081, 0.52497919])

    # Scenario 2:
    x.value = [2.0, 1.0, 0.1]
    layer.forward()
    npt.assert_almost_equal(layer.value, [0.6590011, 0.242433, 0.0985659])

    # If we run this function with larger numbers (or large negative numbers) we have a problem.
    # Compute the softmax of vector x in a numerically stable way:
    x.value = [1000, 2000, 3000]
    layer.forward()
    npt.assert_almost_equal(layer.value, [0.0, 0.0, 1.0])


def test_sigmoid_layer():
    # Topology
    x = Input()
    layer = Sigmoid()(x)

    # Scenario 1:
    x.value = np.array([-10, -0.21, 0, 0.1, 10])
    layer.forward()
    npt.assert_almost_equal(layer.value, [4.539786e-5, 0.44769209, 0.5, 0.52497919, 0.99995460])


def test_sigmoid_layer_derivative():
    # Topology
    x = Input()
    x.value = np.array([0.1, -0.1])
    the_sigmoid = Sigmoid()(x)
    out = Identity()(the_sigmoid)

    # Scenario 1
    # ==========
    # Sigmoid Forward:
    the_sigmoid.forward()
    npt.assert_almost_equal(the_sigmoid.value, [0.5249791875, 0.4750208125])
    # Identity Forward:
    out.forward()
    npt.assert_almost_equal(out.value, [0.5249791875, 0.4750208125])
    # Identity Backprop:
    out.backward()
    npt.assert_almost_equal(out.gradients[the_sigmoid], [1., 1.])
    # Sigmoid Backprop:
    the_sigmoid.backward()
    npt.assert_almost_equal(the_sigmoid.gradients[x], [0.24937604, 0.24937604])


def test_relu_layer():
    # Topology
    x = Input()
    layer = ReLU()(x)

    # Scenario 1:
    x.value = np.array([-10, -0.21, -0.1, 0, 0.1, 0.2, 10])
    layer.forward()
    npt.assert_almost_equal(layer.value, [0, 0, 0, 0, 0.1, 0.2, 10])


def test_relu_layer_derivative():
    # Topology
    x = Input()
    x.value = np.array([0.1, -0.1])
    the_relu = ReLU()(x)
    out = Identity()(the_relu)

    # Scenario 1
    # ==========
    # Relu Forward:
    the_relu.forward()
    npt.assert_almost_equal(the_relu.value, [0.1, 0])
    # Identity Forward:
    out.forward()
    npt.assert_almost_equal(out.value, [0.1, 0])
    # Identity Backprop:
    out.backward()
    npt.assert_almost_equal(out.gradients[the_relu], [1., 1.])
    # Sigmoid Backprop:
    the_relu.backward()
    npt.assert_almost_equal(the_relu.gradients[x], [1, 0])

