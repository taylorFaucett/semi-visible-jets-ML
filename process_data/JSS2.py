import numpy as np

def pT(X):
    return np.sum(np.sum(X, axis=-1), axis=-1)