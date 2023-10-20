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



if __name__ == '__main__':
    main()