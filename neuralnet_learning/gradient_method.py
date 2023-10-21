import numpy as np
import matplotlib.pyplot as plt


def f2(x):
    return np.sum(x ** 2)


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


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    """
    f: The target function that is aimed to be optimised
    init_x: default value of x
    lr: learning rate
    step_num: The total times of executing the gradient method
    """
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)
        x -= lr * grad
    
    return x, np.array(x_history)


def find_min_using_gradient_descent():
    init_x = np.array([-3., 4.])
    x_min, x_history = gradient_descent(f=f2, init_x=init_x, lr=0.1, step_num=100)
    print(x_min)


def draw_gradient_descent():
    init_x = np.array([-3., 4.])
    lr = 0.1
    step_num = 20

    x, x_history = gradient_descent(f2, init_x, lr, step_num)

    plt.plot([-5, 5], [0, 0], '--b')
    plt.plot([0, 0], [-5, 5], '--b')
    plt.plot(x_history[:,0], x_history[:,1], 'o')

    plt.xlim(-3.5, 3.5)
    plt.ylim(-4.5, 4.5)
    plt.xlabel('X0')
    plt.ylabel('X1')
    plt.show()

def bad_example():
    # IF learning rate is set to be too high initially
    init_x = np.array([-3.0, 4.0])
    lr = 10
    x_min, _ = gradient_descent(f2, init_x, lr)
    print(x_min)

    # IF learning rate is set to be too low initially
    init_x = np.array([-3.0, 4.0])
    lr = 1e-10
    x_min, _ = gradient_descent(f2, init_x, lr)
    print(x_min)


if __name__ == "__main__":
    find_min_using_gradient_descent()
    bad_example()