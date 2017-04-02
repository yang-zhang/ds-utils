import unittest
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


class TestMathFunctionsMethods(unittest.TestCase):
    def test_log_inv(self):
        x = np.random.exponential(size=10)
        np.testing.assert_allclose(x, log_inv(log(x)))
        np.testing.assert_allclose(x, log_inv(log(x, 5), 5))

    def test_mad(self):
        x = np.array([1, 2, 3, 5, 7, 8])
        self.assertEqual(mad(x, constant=1), 2.5)


if __name__ == '__main__':
    unittest.main()
