import sys, os
sys.path.append(os.pardir)

import pickle
import numpy as np
from toolz import pipe
from dataset.mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def identity_function(*args):
    return args


def weighted_sum(x, w, b):
    return np.dot(x, w) + b


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    exp_sum = np.sum(exp_a)
    return exp_a / exp_sum


def get_data():
    _, (x_test, t_test) = pipe(
        load_mnist(flatten=True, normalize=True),
        lambda x: identity_function(*x)
    )
    return x_test, t_test
    

def init_network():
    with open("sample_weight.pkl", "rb") as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    y = pipe(
        weighted_sum(x, W1, b1),
        lambda a1: sigmoid(a1),
        lambda z1: weighted_sum(z1, W2, b2),
        lambda a2: sigmoid(a2),
        lambda z2: weighted_sum(z2, W3, b3),
        lambda a3: softmax(a3)
    )
    return y

if __name__ == "__main__":
    x, t = get_data()
    network = init_network()

    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)
        if p == t[i]:
            accuracy_cnt += 1

    print(f"Accuracy: {str(float(accuracy_cnt) / len(x))}")
