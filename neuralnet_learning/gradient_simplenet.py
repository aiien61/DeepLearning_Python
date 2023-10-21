import sys, os
sys.path.append(os.pardir)

import numpy as np
from neural_network.networking import softmax
from loss_functions import cross_entropy_error
from numerical_differentiation import numerical_gradient

class SimpleNet:
    def __init__(self):
        # initialise default weights with normal distribution
        self.W = np.random.randn(2, 3)

    
    def predict(self, x):
        return np.dot(x, self.W)
    

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


if __name__ == "__main__":
    net = SimpleNet()
    print(net.W)

    x = np.array([0.6, 0.9])  # one sample with two features (i.e. two dimensions)
    p = net.predict(x)  # based on this two features of x to predict the class of x
    print(p)
    print(np.argmax(p))

    t = np.array([0, 0, 1])  # corresponding label of x
    print(net.loss(x, t))  # error distance between predicted x and actual t

    f = lambda w: net.loss(x, t)
    dW = numerical_gradient(f, net.W)
    print(dW)
