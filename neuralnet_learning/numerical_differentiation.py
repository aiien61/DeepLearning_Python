import numpy as np
import matplotlib.pyplot as plt


# bad
def numerical_diff_bad(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h

# better
def numerical_diff(f, x):
    h = 1e-4
    return (f(x + h) - f(x - h)) / (2 * h)


def f(x):
    return 0.01 * x ** 2 + 0.1 * x


def show_f_graph():
    x = np.arange(0.0, 20.0, 0.1)
    y = f(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y)
    plt.show()


def tangent_line(f, x, slope):
    y = f(x) - slope * x
    return lambda t: slope * t + y


def main():
    s1 = numerical_diff(f, 5)
    s2 = numerical_diff(f, 10)
    print(s1)
    print(s2)

    x = np.arange(0.0, 20.0, 0.1)
    y = f(x)
    plt.xlabel("x")
    plt.ylabel("f(x)")

    tangent_s1 = tangent_line(f, 5, s1)
    tangent_s2 = tangent_line(f, 10, s2)

    y_s1 = tangent_s1(x)
    y_s2 = tangent_s2(x)

    plt.plot(x, y, label='f(x)')
    plt.plot(x, y_s1, label='tangent of s1', linestyle='dashed')
    plt.plot(x, y_s2, label='tangent of s2', linestyle='dashed')
    plt.legend()
    plt.show()


def f2(x):
    return np.sum(x ** 2)


def f2_tmp1(x0):
    return x0 ** 2 + 4 ** 2


def f2_tmp2(x1):
    return 3 ** 2 + x1 ** 2


def partial():
    print(numerical_diff(f2_tmp1, 3))
    print(numerical_diff(f2_tmp2, 4))


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # gen an array with same size as x

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        i = it.multi_index
        tmp_val = x[i]
        x[i] = tmp_val + h
        fxh1 = f(x)  # f(x + h)

        x[i] = tmp_val - h
        fxh2 = f(x)

        grad[i] = (fxh1 - fxh2) / ( 2 * h)

        x[i] = tmp_val # restore
        it.iternext()

    # --- Only workable for 1-dimensional space ---
    # for i in range(x.size):
    #     tmp_val = x[i]

    #     # f(x + h)
    #     x[i] = tmp_val + h
    #     fxh1 = f(x)

    #     # f(x - h)
    #     x[i] = tmp_val - h
    #     fxh2 = f(x)

    #     grad[i] = (fxh1 - fxh2) / (2 * h)
    #     x[i] = tmp_val  # restore
    
    return grad


def gradient():
    print(numerical_gradient(f2, np.array([3.0, 4.0])))
    print(numerical_gradient(f2, np.array([0.0, 2.0])))
    print(numerical_gradient(f2, np.array([3.0, 0.0])))
    print(numerical_gradient(f2, np.arange(6).reshape(2, 3)))


if __name__ == '__main__':
    gradient()
