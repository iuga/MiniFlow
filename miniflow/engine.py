from miniflow.layers import Input


class Engine(object):
    """
    Core Training Engine
    """
    def topological_sort(self, feed_dict):
        """
        Sort the nodes in topological order using Kahn's Algorithm.

        Arguments:
            `feed_dict`: A dictionary where the key is a `Input` Node and the value is the respective
                         value feed to that Node.
            Returns a list of sorted nodes.
        """
        input_nodes = [n for n in feed_dict.keys()]

        G = {}
        nodes = [n for n in input_nodes]
        while len(nodes) > 0:
            n = nodes.pop(0)
            if n not in G:
                G[n] = {'in': set(), 'out': set()}
            for m in n.outbounds:
                if m not in G:
                    G[m] = {'in': set(), 'out': set()}
                G[n]['out'].add(m)
                G[m]['in'].add(n)
                nodes.append(m)

        L = []
        S = set(input_nodes)
        while len(S) > 0:
            n = S.pop()

            if isinstance(n, Input):
                n.value = feed_dict[n]

            L.append(n)
            for m in n.outbounds:
                G[n]['out'].remove(m)
                G[m]['in'].remove(n)
                # if no other incoming edges add to S
                if len(G[m]['in']) == 0:
                    S.add(m)
        return L

    def forward(self, graph):
        """
        Performs a forward pass through a list of sorted Nodes and return the results.

        Arguments:
            `graph`: The result of calling `topological_sort`.
        """
        # Forward pass
        for n in graph:
            n.forward()
        return graph[-1].value

    def forward_and_backward(self, graph):
        """
        Performs a forward pass and a backward pass through a list of sorted Nodes.
        Every node in the network get some inputs and can right awas compute two things:
        - The output value: forward()
        - The local gradient of its inputs with respect to its ouput value: backward()
        Notice that the layers can do this completely independently without being aware of
        any of the details of the full network that they are part. However, once the forward
        pass is over, during backpropagation the layer will learn about the gradient of it's
        ouput value on the final output of the network. The chain rule says that the layer
        should take the gradient and multiply it into every gradient it normally computes for
        all of its inputs.

        Chain Rule
        ----------
        This expresion:
            f(x, y, z) = (x + y) * z
        can be broken down into two expresions:
            - q = (x + y)
            - f = q * z
        The chain rule tells us that the correct way to "chain" these gradients expressions
        together is through multiplication. It could be visualized through nodes:

            [x]+-+
                 +->[+]+
            [y]+-+     |
                       +->[*]+> = (x+y)*z
            [z]+-------+

        Arguments:
            `graph`: The result of calling `topological_sort`.
        """
        # Forward pass
        for n in graph:
            n.forward()
        # Backward pass
        for n in graph[::-1]:
            n.backward()

    def sgd_update(self, trainables, learning_rate=1e-2):
        """
        Process of finding the set of parameters W,b,etc that minimize the loss function.
        The gradients tell us the direction in which the function has the steepest rate of
        increase, but it does not tell us how far along this direction should be.
        Choosing the step size ( learining rate ) will become one of the most important
        hyperaparameters settings in training a neural network.

        This method updates the value of each trainable with Stochastic Gradient Descent.

        The gradient of a mini-batch is a good aproximation of the gradient of the full objective.

        Vanilla Gradient Descent:
        This simple loop is the core of all Neural Network libraries:
        >>> while True:
        >>>     # new_params -= size * direction
        >>>     weights += - step_size * weights_gradient

        Arguments:
            `trainables`: A list of `Input` Nodes representing weights/biases.
            `learning_rate`: The learning rate.
        """
        # Loop over the trainables
        for trainable in trainables:
            # Change the trainable's value by subtracting the learning rate
            # multiplied by the partial of the cost with respect to this trainable.
            partial = trainable.gradients[trainable]
            trainable.value -= learning_rate * partial
