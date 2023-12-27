import sys, os
sys.path.append(os.pardir)

import dataset.load_mnist as lm

import numpy as np
import neuralnet as nl

dataset = lm.load_mnist()
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

print(y_test[0: 10])


w_list, b_list = nl.make_params([784, 100, 10])


for epoch in range(100):
    ra = np.random.randint(60_000, size=60_000)
    for i in range(60):
        x_batch = x_train[ra[i * 1000 : (i+1) * 1000], :]
        y_batch = y_train[ra[i * 1000 : (i+1) * 1000], :]
        w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, eta=2.0)

# evaluation
val_dict = nl.calculate(x_test, w_list, b_list)
print(val_dict['y_2'][0: 10].round(2))