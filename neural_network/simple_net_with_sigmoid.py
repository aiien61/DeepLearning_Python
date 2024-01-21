import numpy as np
from toolz import pipe


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identity_function(x):
    return x


def weighted_sum(x, w, b):
    return np.dot(x, w) + b


def init_network():
    network = {}
    network['W1'] = np.array([[.1, .3, .5], [.2, .4, .6]])
    network['b1'] = np.array([.1, .2, .3])
    network['W2'] = np.array([[.1, .4], [.2, .5], [.3, .6]])
    network['b2'] = np.array([.1, .2])
    network['W3'] = np.array([[.1, .3], [.2, .4]])
    network['b3'] = np.array([.1, .2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    y = pipe(
        weighted_sum(x, W1, b1),
        lambda a1: sigmoid(a1),
        lambda z1: weighted_sum(z1, W2, b2),
        lambda a2: sigmoid(a2),
        lambda z2: weighted_sum(z2, W3, b3),
        lambda a3: identity_function(a3)
    )

    return y

if __name__ == "__main__":
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)

    