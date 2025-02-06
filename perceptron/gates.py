import numpy as np
from icecream import ic

class Gate:
    """
    linear partition: AND, NAND, OR
    non-linear partition: XOR
    """

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
    
    @classmethod
    def AND(cls, x1: int, x2: int) -> int:
        X = np.array([x1, x2])
        W = np.array([0.5, 0.5])
        b = -0.7
        return 0 if np.sum(X * W) + b <= 0 else 1
    
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
    
    @classmethod
    def NAND(cls, x1: int, x2: int) -> int:
        X = np.array([x1, x2])
        W = np.array([-0.5, -0.5])
        b = 0.7
        return 0 if np.sum(X * W) + b <= 0 else 1


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

    @classmethod
    def OR(cls, x1: int, x2: int) -> int:
        X = np.array([x1, x2])
        W = np.array([0.5, 0.5])
        b = -0.2
        return 0 if np.sum(X * W) + b <= 0 else 1


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

    #              NAND OR     AND
    #   x1  x2  |   s1  s2  |   y
    # -----------------------------
    #   0   0   |   1   0   |   0
    # -----------------------------
    #   1   0   |   1   1   |   1
    # -----------------------------
    #   0   1   |   1   1   |   1
    # -----------------------------
    #   1   1   |   0   1   |   0

    @classmethod
    def XOR(cls, x1: int, x2: int) -> int:
        s1 = cls.NAND(x1, x2)
        s2 = cls.OR(x1, x2)
        y = cls.AND(s1, s2)
        return y


def main():
    inputs = [(0, 0), (1, 0), (0, 1), (1, 1)]
    gates = {'AND': Gate.AND, 'NAND': Gate.NAND, 'OR': Gate.OR, 'XOR': Gate.XOR}

    for xs in inputs:
        ic(xs)
        for gate, f in gates.items():
            print(f"{gate}{xs}: {f(*xs)}")
        print()

if __name__ == "__main__":
    main()
