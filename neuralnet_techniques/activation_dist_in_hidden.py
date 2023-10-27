import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def tanh(x):
    return np.tanh(x)


if __name__ == "__main__":
    x = np.random.randn(1000, 100)  # size of input data
    node_num = 100  # number of nodes on each layer
    hidden_layer_size = 5  # 5 layers
    activations = {}  # for creating the result of activation

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i - 1]
        
        # std = 1
        # w = np.random.randn(node_num, node_num) * 1
        
        # std = 1
        # w = np.random.rand(node_num, node_num) * 0.01

        # Xavier's default
        w = np.random.rand(node_num, node_num) * np.sqrt(1 / node_num)

        a = np.dot(x, w)
        
        z = sigmoid(a)
        # z = relu(a)
        # z = tanh(a)
        activations[i] = z
    
    for i, a in activations.items():
        plt.subplot(1, len(activations), i+1)
        plt.title(f"{i+1}-layer")
        if i != 0:
            plt.yticks([], [])
        plt.hist(a.flatten(), 30, range=(0, 1))
    plt.show()
