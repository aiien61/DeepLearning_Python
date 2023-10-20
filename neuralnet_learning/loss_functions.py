import numpy as np

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


def main():
    # one-hot encoding labels, e.g. correct answer is 2
    t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # output of softmax.
    # if 2 has highest probability 0.6
    y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]

    mse_good_guess = mean_squared_error(np.array(y), np.array(t))
    entropy_good_guess = cross_entropy_error(np.array(y), np.array(t))

    # if 7 has highest probability 0.6
    y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]

    mse_bad_guess = mean_squared_error(np.array(y), np.array(t))
    entropy_bad_guess = cross_entropy_error(np.array(y), np.array(t))

    print(f'mse of the good guess: {mse_good_guess}')
    print(f'mse of the bad guess: {mse_bad_guess}')
    print(f'error of the good guess is smaller than bad guess: {mse_good_guess < mse_bad_guess}')
    
    print(f'cross entropy error of the good guess: {entropy_good_guess}')
    print(f'cross entropy error of the bad guess: {entropy_bad_guess}')
    print(f'error of the good guess is smaller than bad guess: {entropy_good_guess < entropy_bad_guess}')


if __name__ == "__main__":
    main()