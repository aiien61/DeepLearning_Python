import sys, os
sys.path.append(os.pardir)

import pickle
import numpy as np
from toolz import pipe
from dataset.mnist import load_mnist


def identity_function(*args):
    return args


def get_data():
    (x_train, t_train), (x_test, t_test) = pipe(
        load_mnist(normalize=True, one_hot_label=True),
        lambda x: identity_function(*x)
    )
    return (x_train, t_train), (x_test, t_test)


def cross_entropy_error(y, t, is_ont_hot: bool):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]

    if is_ont_hot:
        return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
    else:
        return -np.sum(t * np.log(y)) / batch_size


if __name__ == "__main__":
    # (x_train, t_train), (x_test, t_test) = get_data()

    # print(x_train.shape)
    # print(t_train.shape)
    # print(x_test.shape)
    # print(t_test.shape)
    print(np.random.choice(60_000, 10))
