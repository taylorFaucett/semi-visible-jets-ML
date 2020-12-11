import numpy as np


def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x - mean) / std


def process_data(X):
    X = np.log(1.0 + X) / 4.0
    X = scale_data(X)
    return X
