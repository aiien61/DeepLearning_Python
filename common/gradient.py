import numpy as np


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # gen an array with same size as x

    for i in range(x.size):
        tmp_val = x[i]

        # f(x + h)
        x[i] = tmp_val + h
        fxh1 = f(x)

        # f(x - h)
        x[i] = tmp_val - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / (2 * h)
        x[i] = tmp_val  # restore
    
    return grad