import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def relu(x):
    return np.maximum(0, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
    return y
