from miniflow.engine import Engine
from miniflow.layers import Input
from sklearn.utils import resample
from miniflow.losses import MSE
from click import progressbar
from sys import stdout


class Model(object):
    """
    Define a generic Network model topology and provide methods to train the Network
    """
    losses = {
        'mse': MSE
    }
    optimizers = {
        'sgd': 'sgd'
    }

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.optimizer = None
        self.loss = None
        self.engine = Engine()

    def compile(self, optimizer='sdg', loss='mse'):
        """
        Compile the model
        """
        # Create the Loss function as a layer:
        self.loss = self.losses.get(loss, MSE)

    def train(self, X_train, y_train, feed_dict={}, X_test=None, y_test=None, epochs=1000, batch_size=128, learning_rate=0.01):
        """
        Try to reach the bottom of the loss function as a high-dimensional optimization landscape:

        We motivated the idea of optimizing the loss function with iterative refinement, where we start
        with a random set of weights and refine them step by step until the loss is minimized.

        This method performs a Mini-Batch Gradient Descent:
        In large-scale applications the training data can have millions of examples, Hence, it seems wasteful
        to compute the full loss function over the entire training set in order to perform a single parameter
        update. A very common approach to addressing this challenge is to compute the gradient over batches
        of training data.

        Parameters:
            `X_train`<List>: The data used to create the minibatches
            `y_train`<List>: The correct labels for each example in X_train
            `feed_dict`<Dict>: Dictionary with the placeholders and the data to use in them.
            `X_test`<List>: The data used to validate the minibatches
            `y_test`<List>: The correct labels for each example in X_test
            `epochs`<int>: The numbers of epochs
            `batch_size`<int>: The size of the mini-batch used to train
            `learning_rate`<float>: The step size to update the trainables.
        """
        # Calculate the number of mini-batches per epoch:
        steps_per_epoch = X_train.shape[0] // batch_size
        # Sort the network graph
        self.graph = self.engine.topological_sort(feed_dict)
        # Crete the Network Inputs:
        y_input_value = Input(trainable=False, name="y_input")
        # Add the loss layer in the graph:
        loss_layer = self.loss(y_input_value)(self.outputs)
        self.graph.append(loss_layer)
        # Get all the trainables
        trainables = []
        for layer in feed_dict.keys():
            if layer.trainable:
                trainables.append(layer)
        # Create the history object to record all data:
        history = {
            'train_loss': [],
            'test_loss': []
        }
        # Train the model:
        for i in range(epochs):
            loss = 0
            test_loss = 0
            with progressbar(range(steps_per_epoch), label='Epoch {}: '.format(i + 1), file=stdout) as bar:
                for mini_batch in bar:
                    # Step 1
                    # Randomly sample a batch of examples:
                    X_batch, y_batch = resample(X_train, y_train, n_samples=batch_size)
                    # Reset value of X and y Inputs
                    for input in self.inputs:
                        input.value = X_batch
                    y_input_value.value = y_batch

                    # Step 2
                    self.engine.forward_and_backward(self.graph)

                    # Step 3
                    self.engine.sgd_update(trainables, learning_rate=learning_rate)

                    # Step 4
                    loss += self.graph[-1].value

                    # Measure the results in a test set:
                    if X_test is not None and y_test is not None:
                        for input in self.inputs:
                            input.value = X_test
                        y_input_value.value = y_test
                        ouput = self.engine.forward(self.graph)
                        test_loss += ouput

            history['train_loss'].append(loss / steps_per_epoch)
            history['test_loss'].append(test_loss / steps_per_epoch)
            print("Train Loss: {:.3f} - Test Loss: {:.3f}".format(loss / steps_per_epoch, test_loss / steps_per_epoch))

        return history

    def summary(self):
        print("Summary:")
