import numpy as np
from activation_functions import sigmoid

# Input layer -> first layer
# input layer: nodes x1, x2, bias 1
# first layer: nodes Z1

X = np.array([1, 0.5])
W1 = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape)
print(X.shape)
print(B1.shape)

A1 = np.matmul(X, W1.T) + B1
print(A1)

Z1 = sigmoid(A1)
print(Z1)

# first layer -> second layer
# input layer: nodes Z1, bias 1
# first layer: nodes Z2

W2 = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
B2 = np.array([0.1, 0.2])
print(Z1.shape)
print(W2.shape)
print(B2.shape)

A2 = np.matmul(Z1, W2.T) + B2
print(A2)
Z2 = sigmoid(A2)
print(Z2)

# second layer -> third layer
# input layer: nodes Z2, bias 1
# first layer: nodes Y

def identity_function(x):
    return x

W3 = np.array([[0.1, 0.2], [0.3, 0.4]])
B3 = np.array([0.1, 0.2])
print(Z2.shape)
print(W3.shape)
print(B3.shape)

A3 = np.matmul(Z2, W3.T) + B3
print(A3)
Z3 = identity_function(A3)
print(Z3)


def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.2], [0.3, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.matmul(x, W1.T) + b1
    z1 = sigmoid(a1)
    a2 = np.matmul(z1, W2.T) + b2
    z2 = sigmoid(a2)
    a3 = np.matmul(z2, W3.T) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)