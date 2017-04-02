from miniflow.engine import topological_sort, forward_and_backward, sgd_update
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

    def train(self, X_train, y_train, Xi, yi, feed_dict, epochs=1000, batch_size=128):
        # Total number of examples
        steps_per_epoch = X_train.shape[0] // batch_size
        # Sort the graph
        self.graph = topological_sort(feed_dict)
        # Add the loss layer in the graph:
        loss_layer = self.loss(yi)(self.outputs)
        self.graph.append(loss_layer)
        # Get all the trainables
        trainables = []
        for layer in feed_dict.keys():
            if layer.trainable:
                trainables.append(layer)

        # Train the model:
        for i in range(epochs):
            loss = 0
            with progressbar(range(steps_per_epoch), label='Training: ') as bar:
                for j in bar:
                    # Step 1
                    # Randomly sample a batch of examples
                    X_batch, y_batch = resample(X_train, y_train, n_samples=batch_size)

                    # Reset value of X and y Inputs
                    Xi.value = X_batch
                    yi.value = y_batch

                    # Step 2
                    forward_and_backward(self.graph)

                    # Step 3
                    sgd_update(trainables, learning_rate=0.05)

                    loss += self.graph[-1].value
            print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))

    def summary(self):
        print("Summary:")
