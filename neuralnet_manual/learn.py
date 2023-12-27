import sys, os
sys.path.append(os.pardir)
import dataset.load_mnist as lm

import numpy as np
import neuralnet as nl
import matplotlib.pyplot as plt

dataset = lm.load_mnist()
x_train = dataset['x_train']
y_train = dataset['y_train']
x_test = dataset['x_test']
y_test = dataset['y_test']

w_list, b_list = nl.make_params([784, 100, 10])

train_time = 10_000
batch_size = 1_000
total_acc_list = []
total_loss_list = []

for epoch in range(100):
    ra = np.random.randint(60_000, size=60_000)
    for i in range(60):
        x_batch = x_train[ra[i * batch_size: (i+1) * batch_size], :]
        y_batch = y_train[ra[i * batch_size: (i+1) * batch_size], :]
        w_list, b_list = nl.update(x_batch, w_list, b_list, y_batch, eta=2.0)

    acc_list = []
    loss_list = []

    # auto evaluation
    for k in range(train_time // batch_size):
        x_batch = x_test[k * batch_size: (k+1) * batch_size, :]
        y_batch = y_test[k * batch_size: (k+1) * batch_size, :]

        acc_val = nl.accuracy(x_batch, w_list, b_list, y_batch)
        loss_val = nl.loss(x_batch, w_list, b_list, y_batch)

        acc_list.append(acc_val)
        loss_list.append(loss_val)

    acc = np.mean(acc_list)
    loss = np.mean(loss_list)

    total_acc_list.append(acc)
    total_loss_list.append(loss)
    print(f"epoch: {epoch}, Accuracy: {acc}, Loss: {loss}")

# manual evaluation
print(y_test[0: 10])
val_dict = nl.calculate(x_test, w_list, b_list, y_test)
print(val_dict['y_2'][0: 10].round(2))

plt.subplot(211)
plt.plot(np.arange(0, len(total_acc_list)), total_acc_list)
plt.title('accuracy')
plt.subplot(212)
plt.plot(np.arange(0, len(total_acc_list)), total_loss_list)
plt.title('loss')
plt.tight_layout()
plt.show()
