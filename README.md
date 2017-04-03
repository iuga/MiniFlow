# MiniFlow
It's a Small Deep Learning framework based on NymPy. The only purpose it has is learn how networks work internally. The code is extremelly well documented and it's easy to follow.

## Instalation
```bash
# conda is required
./tools/install.sh
```

## Examples:
```python
# Neural network inputs (X and y):
Xi = Input(name="X_input")
yi = Input(name="y_input")

# Neural Network trainable parameter:
W1i, b1i = Variable(name="W1"), Variable(name="b1")
W2i, b2i = Variable(name="W2"), Variable(name="b2")

# Define the Network Topology
Xi = Input()
x = Linear(W1i, b1i)(Xi)
x = Sigmoid()(x)
x = Linear(W2i, b2i)(x)

# Define the model:
model = Model(inputs=[Xi], outputs=[x])
# Compile the model
model.compile(loss='mse')
model.train(X, y, epochs=1000, batch_size=32, feed_dict = {
    W1i: W1,
    b1i: b1,
    W2i: W2,
    b2i: b2
})
```
```bash
Training:   [####################################]  100%
Epoch: 1, Loss: 864.557
Training:   [####################################]  100%
Epoch: 2, Loss: 24.480
Training:   [####################################]  100%
Epoch: 3, Loss: 20.955
...
Training:   [####################################]  100%
Epoch: 1000, Loss: 0.219
```
