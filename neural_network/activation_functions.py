import numpy as np
import matplotlib.pyplot as plt

# step function: determine outcome by the comparison with threshold
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0

# refactor to be compatible with numpy
def step_function_numpy(x):
    y = x > 0
    return y.astype(int)


def plot_step_function():
    def step_function(x):
        return np.array(x > 0, dtype=np.int64)
    
    x = np.arange(-5, 5, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_sigmoid():
    x = np.arange(-5, 5, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)
    plt.show()


def plot_step_function_and_sigmoid():
    x = np.arange(-5, 5, 0.1)
    y_sigmoid = sigmoid(x)
    y_step = step_function_numpy(x)
    plt.plot(x, y_sigmoid, label='sigmoid')
    plt.plot(x, y_step, label='step function', linestyle='dashed')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()


# ReLU: Rectified Linear Unit
def relu(x):
    return np.maximum(0, x)


def plot_relu():
    x = np.arange(-5, 5, 0.1)
    y = relu(x)
    plt.plot(x, y)
    plt.ylim(-1, 5.1)
    plt.show()


if __name__ == '__main__':
    plot_relu()