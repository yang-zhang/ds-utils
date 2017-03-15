import numpy as np


def log(x, base=None):
    if base:
        result = np.log(x) / log(base)
    else:
        result = np.log(x)
    return result


def log_inv(x, base=None):
    if base:
        result = np.power(base, x)
    else:
        result = np.exp(x)
    return result


# median absolute deviation
def mad(x, constant=1.4826):
    return constant * np.median(abs(x - np.median(x)))


if __name__ == '__main__':
    x = np.random.exponential(size=10)
    assert np.allclose(x, log_inv(log(x)))
    assert np.allclose(x, log_inv(log(x, 5), 5))

    x = np.array([1, 2, 3, 5, 7, 8])
    assert mad(x, constant=1) == 2.5
