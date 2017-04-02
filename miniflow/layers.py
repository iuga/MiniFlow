import numpy as np


class Layer(object):
    """
    Base class for nodes in the network.
    Arguments:
        `inbounds`: A list of layers with edges into this node.
    """
    def __init__(self, inbounds=[]):
        """
        Layer's constructor (runs when the object is instantiated). Sets
        properties that all layers need.
        """
        # The eventual value of this node. Set by running the forward() method.
        self.value = None
        # A list of layers that this layers inputs to:
        self.inbounds = []
        # A list of layers that this layer outputs to.
        self.outbounds = []
        # Keys are the inputs to this node and
        # their values are the partials of this node with
        # respect to that input.
        self.gradients = {}

    def __call__(self, inbounds=[]):
        """
        Call the Layer object with the inbound nodes and create the
        links between them. The logic should be in call()
        """
        return self.call(inbounds=inbounds)

    def call(self, inbounds=[]):
        """
        Call the layer with the inbounds nodes ( previous layers in the net )
        """
        # A list of layers with edges into this layer.
        inbounds = inbounds if isinstance(inbounds, list) else [inbounds]
        for inbound in inbounds:
            self.inbounds.insert(0, inbound)
        # Sets this layer as an outbound layer for all of this layers's inputs.
        # Link between this layer and the previous ones. In other words, fill the
        # Outbounds from the previous layer with this one ( Create the connection ).
        for layer in self.inbounds:
            layer.outbounds.append(self)
        return self

    def forward(self):
        """
        Every layer that uses this class as a base class will
        need to define its own `forward` method.
        """
        raise NotImplementedError

    def backward(self):
        """
        Every layer that uses this class as a base class will
        need to define its own `backward` method.
        """
        raise NotImplementedError


class Input(Layer):
    """
    A generic input into the network.
    """
    def __init__(self):
        # The base class constructor has to run to set all
        # the properties here.
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        Layer.__init__(self)

    def forward(self):
        # Do nothing because nothing is calculated.
        pass

    def backward(self):
        # An Input node has no inputs so the gradient (derivative) is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {
            self: 0
        }
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbounds:
            self.gradients[self] += n.gradients[self]


class Linear(Layer):
    """
    Represents a node that performs a linear transform.
    """
    def __init__(self, W, b):
        # The base class constructor. Weights and bias
        # are treated like inbound nodes.
        Layer.__init__(self)
        self.W = W
        self.b = b
        self.inbounds.append(W)
        self.inbounds.append(b)

    def forward(self):
        """
        Performs the math behind a linear transform.
        """
        X = self.inbounds[0].value
        W = self.W.value
        b = self.b.value
        self.value = np.dot(X, W) + b

    def backward(self):
        """
        Calculates the gradient based on the output values.
        """
        # Initialize a partial for each of the inbound_nodes.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbounds}
        self.gradients[self.W] = np.zeros_like(self.W.value)
        self.gradients[self.b] = np.zeros_like(self.b.value)
        # Cycle through the outputs. The gradient will change depending
        # on each output, so the gradients are summed over all outputs.
        for n in self.outbounds:
            # Get the partial of the cost with respect to this node.
            grad_cost = n.gradients[self]
            # Set the partial of the loss with respect to this node's inputs.
            self.gradients[self.inbounds[0]] += np.dot(grad_cost, self.W.value.T)
            # Set the partial of the loss with respect to this node's weights.
            self.gradients[self.W] += np.dot(self.inbounds[0].value.T, grad_cost)
            # Set the partial of the loss with respect to this node's bias.
            self.gradients[self.b] += np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Layer):
    """
    Represents a node that performs the sigmoid activation function.
    """
    def __init__(self):
        # The base class constructor.
        Layer.__init__(self)

    def _sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.inbounds[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of
        the sigmoid function.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbounds}
        # Sum the partial with respect to the input over all the outputs.
        for n in self.outbounds:
            grad_cost = n.gradients[self]
            sigmoid = self.value
            self.gradients[self.inbounds[0]] += sigmoid * (1 - sigmoid) * grad_cost
