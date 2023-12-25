import sys, os
sys.path.append(os.pardir)

from dataset.mnist import load_mnist

import numpy as np
import neuralnet as nl

(x_train, y_train), (x_test, y_test) = load_mnist()
w_list, b_list = nl.make_params([784, 100, 10])

for epoch in range(1):
    ra = np.random.randint(60_000, size=60_000)
    for i in range(60):
        x_batch = x_train[ra[i * 1000 : (i+1) * 1000], :]
        y_batch = y_train[ra[i * 1000 : (i+1) * 1000], :]
        w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, eta=2.0)
