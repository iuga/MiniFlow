from miniflow.engine import topological_sort, forward_and_backward, sgd_update
from miniflow.layers import Input
from sklearn.utils import resample
from miniflow.losses import MSE
from click import progressbar


class Model(object):
    """
    Define a generic Network model topology
    """
    losses = {
        'mse': MSE
    }

    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        self.optimizer = None
        self.loss = None

    def compile(self, optimizer='sdg', loss='mse'):
        """
        Compile the model
        """
        # Create the Loss function as a layer:
        self.loss = self.losses.get(loss, MSE)

    def train(self, X_train, y_train, feed_dict, epochs=1000, batch_size=128):
        # Total number of examples
        steps_per_epoch = X_train.shape[0] // batch_size
        # Sort the graph
        self.graph = topological_sort(feed_dict)
        # Network Inputs:
        y_train_input = Input(trainable=False, name="y_input")
        # Add the loss layer in the graph:
        loss_layer = self.loss(y_train_input)(self.outputs)
        self.graph.append(loss_layer)
        # Get all the trainables
        trainables = []
        for layer in feed_dict.keys():
            if layer.trainable:
                trainables.append(layer)
        # Train the model:
        for i in range(epochs):
            loss = 0
            with progressbar(range(steps_per_epoch), label='Epoch {}: '.format(i + 1)) as bar:
                for j in bar:
                    # Step 1
                    # Randomly sample a batch of examples
                    X_batch, y_batch = resample(X_train, y_train, n_samples=batch_size)

                    # Reset value of X and y Inputs
                    for input in self.inputs:
                        input.value = X_batch
                    y_train_input.value = y_batch

                    # Step 2
                    forward_and_backward(self.graph)

                    # Step 3
                    sgd_update(trainables, learning_rate=0.05)

                    loss += self.graph[-1].value
            print("Train Loss: {:.3f}".format(loss / steps_per_epoch))

    def summary(self):
        print("Summary:")
