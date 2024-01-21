import numpy as np
from toolz import pipe

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a


if __name__ == "__main__":
    print(softmax(np.array([1010, 1000, 990])))