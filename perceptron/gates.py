import numpy as np

# linear partition: AND, NAND, OR

# AND Gate
#   x1  x2  |   y
# -----------------
#   0   0   |   0
# -----------------
#   1   0   |   0
# -----------------
#   0   1   |   0
# -----------------
#   1   1   |   1


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7  # bias
    if np.dot(x, w) + b <= 0:
        return 0
    else:
        return 1

# NAND Gate
#   x1  x2  |   y
# -----------------
#   0   0   |   1
# -----------------
#   1   0   |   1
# -----------------
#   0   1   |   1
# -----------------
#   1   1   |   0

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    if np.dot(x, w) + b <= 0:
        return 0
    else:
        return 1

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

# non-linear partition: XOR

# XOR Gate
#   x1  x2  |   y
# -----------------
#   0   0   |   0
# -----------------
#   1   0   |   1
# -----------------
#   0   1   |   1
# -----------------
#   1   1   |   0

#   x1  x2  |   s1  s2  |   y
# -----------------------------
#   0   0   |   1   0   |   0
# -----------------------------
#   1   0   |   1   1   |   1
# -----------------------------
#   0   1   |   1   1   |   1
# -----------------------------
#   1   1   |   0   1   |   0

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1,x2)
    y = AND(s1, s2)
    return y


def main():
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    gates = {'AND Gate': AND, 'NAND Gate': NAND, 'OR Gate': OR, 'XOR Gate': XOR}
    for gate, f in gates.items():
        print(gate)
        for xs in inputs:
            print(f(*xs))

if __name__ == "__main__":
    main()
