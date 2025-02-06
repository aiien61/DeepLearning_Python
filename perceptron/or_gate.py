import numpy as np

# OR Gate
#   x1  x2  |   y
# -----------------
#   0   0   |   0
# -----------------
#   1   0   |   1
# -----------------
#   0   1   |   1
# -----------------
#   1   1   |   1


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    if np.dot(x, w) + b <= 0:
        return 0
    else:
        return 1
