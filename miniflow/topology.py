from miniflow.engine import topological_sort, forward_and_backward, sgd_update
from sklearn.utils import resample
from miniflow.losses import MSE


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

    def compile(self, optimizer='sdg', loss='rmse'):
        """
        Compile the model
        """
        self.loss = self.losses.get(loss, MSE)
        print("Loss", self.loss)

    def train(self, X_train, y_train, Xi, yi, feed_dict, epochs=1000, batch_size=128, m=0):
        # Total number of examples
        steps_per_epoch = m // batch_size
        # Sort the graph
        graph = topological_sort(feed_dict)
        # Add the loss layer
        loss = self.loss(yi)(self.outputs)
        graph.append(loss)
        # Get all the trainables
        trainables = []
        for layer in feed_dict.keys():
            print("Layer:", layer.name, layer, layer.__class__.__name__, layer.trainable)
            if layer.trainable:
                trainables.append(layer)
        print(trainables)

        print("Total number of examples = {}".format(m))

        # Step 4
        for i in range(epochs):
            loss = 0
            for j in range(steps_per_epoch):
                # Step 1
                # Randomly sample a batch of examples
                X_batch, y_batch = resample(X_train, y_train, n_samples=batch_size)

                # Reset value of X and y Inputs
                Xi.value = X_batch
                yi.value = y_batch

                # Step 2
                forward_and_backward(graph)

                # Step 3
                sgd_update(trainables, learning_rate=0.05)

                loss += graph[-1].value
            print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / steps_per_epoch))

    def summary(self):
        print("Summary:")
