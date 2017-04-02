class Model(object):
    """
    Define a generic Network model topology
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def train(self):
        pass
