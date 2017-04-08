import numpy as np


class Layer(object):
    """
    Base class for nodes in the network.
    Arguments:
        `inbounds`: A list of layers with edges into this node.
    """
    def __init__(self, inbounds=[], trainable=True, name=None):
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
        # Is the layer trainable?
        self.trainable = trainable
        # Set a name, It's really cool for debugging
        self.name = name

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
    def __init__(self, *args, **kwargs):
        # The base class constructor has to run to set all
        # the properties here.
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        Layer.__init__(self, *args, **kwargs)
        self.trainable = False

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


class Variable(Input):
    """
    A generic Input layer that is trainable by default.
    It should be used for internal parameters that should be modified..
    """
    def __init__(self, *args, **kwargs):
        """
        Force the trainable property
        """
        Input.__init__(self, *args, *kwargs)
        self.trainable = True


class Identity(Layer):
    """
    An Identity Layer.
    """
    def __init__(self, *args, **kwargs):
        # The base class constructor has to run to set all
        # the properties here.
        # The most important property on an Input is value.
        # self.value is set during `topological_sort` later.
        Layer.__init__(self, *args, **kwargs)
        self.trainable = False

    def forward(self):
        # Do nothing because nothing is calculated.
        self.value = self.inbounds[0].value

    def backward(self):
        # An Input node has no inputs so the gradient (derivative) is zero.
        # The key, `self`, is reference to this object.
        self.gradients = {n: np.ones_like(n.value) for n in self.inbounds}
        # Weights and bias may be inputs, so you need to sum
        # the gradient from output gradients.
        for n in self.outbounds:
            self.gradients[self] += n.gradients[self]


class Linear(Layer):
    """
    Represents a node that performs a linear transform.
    """
    def __init__(self, W, b, *args, **kwargs):
        # The base class constructor. Weights and bias
        # are treated like inbound nodes.
        Layer.__init__(self, *args, **kwargs)
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
    def __init__(self, *args, **kwargs):
        # The base class constructor.
        Layer.__init__(self, *args, **kwargs)

    def sigmoid(self, x):
        """
        This method is separate from `forward` because it
        will be used with `backward` as well.

        `x`: A numpy array-like object.
        """
        return 1. / (1. + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Compute the sigmoid derivative
        `x`: A numpy array-like object.
        """
        return x * (1. - x) 
    
    def forward(self):
        """
        Perform the sigmoid function and set the value.
        """
        input_value = self.inbounds[0].value
        self.value = self.sigmoid(input_value)

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
            self.gradients[self.inbounds[0]] += self.sigmoid_derivative(self.value) * grad_cost


class ReLU(Layer):
    """
    Rectified Linear Unit
    Represents a node that performs the ReLU activation function.
    In the context of artificial neural networks, the rectifier is an activation function 
    defined as max(0, x) where x is the input to a neuron.  
    """
    def __init__(self, *args, **kwargs):
        # The base class constructor.
        Layer.__init__(self, *args, **kwargs)

    def relu(self, x):
        """
        Basically, it sets anything less than or equal to 0 (negative numbers) to be 0. 
        And keeps all the same values for any values > 0.
        `x`: A numpy array-like object.
        """
        # ReLU returns x if x>0, else 0: x * (x > 0)
        # np.maximum(x, 0, x)
        return x * (x > 0)
    
    def relu_derivative(self, x):
        """
        A "derivative" is just the slope of the graph at certain point. 
        So what is the slope of the graph at the point x=2? 1!
        This holds everywhere > 0, the slope is 1.
        `x`: A numpy array-like object.
        """
        # ReLU returns 1 if x>0, else 0: 1 * (x > 0)
        return 1 * (x > 0)

    def forward(self):
        """
        Perform the ReLU function and set the value.
        """
        input_value = self.inbounds[0].value
        self.value = self.relu(input_value)

    def backward(self):
        """
        Calculates the gradient using the derivative of the ReLU function.
        Now just looking at the equation  f(x)=max(0,x), it was not clear to me what the derivative is.
        Everywhere > 0, the slope is 1.
        """
        # Initialize the gradients to 0.
        self.gradients = {n: np.zeros_like(n.value) for n in self.inbounds}
        # Sum the partial with respect to the input over all the outputs.
        for n in self.outbounds:
            grad_cost = n.gradients[self]
            self.gradients[self.inbounds[0]] += self.relu_derivative(self.value) * grad_cost

class Softmax(Layer):
    """
    Used to predict if some input belongs to one of many classes (classification problem).
    The softmax function squashes the outputs of each unit to be between 0 and 1, just like a sigmoid.
    It also divides each output such that the total sum of the outputs is equal to 1.
    The output of the softmax function is equivalent to a categorical probability distribution, it tells
    you the probability that any of the classes are true (softmax normalizes the outputs so that they sum to one).
    Also, we need to consider the numerical stability.

    Links and Resources:
    http://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    """
    def __init__(self, *args, **kwargs):
        # The base class constructor.
        Layer.__init__(self, *args, **kwargs)

    def forward(self):
        """
        Compute the softmax of vector x in a numerically stable way:
        """
        x = self.inbounds[0].value
        shiftx = x - np.max(x)
        exps = np.exp(shiftx)
        self.value = exps / np.sum(exps)

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
            x = self.value
            # TODO: Define the derivative
            self.gradients[self.inbounds[0]] +=  grad_cost
