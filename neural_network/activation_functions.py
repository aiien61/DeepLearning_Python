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


if __name__ == '__main__':
    plot_step_function()