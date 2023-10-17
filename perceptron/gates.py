def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    if (x1 * w1 + x2 * w2) >= theta:
        return 0
    else:
        return 1


def main():
    print(AND(0, 0))
    print(AND(1, 0))
    print(AND(0, 1))
    print(AND(1, 1))


if __name__ == "__main__":
    main()
