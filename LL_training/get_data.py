import h5py
from sklearn import preprocessing
from process_data import process_data
import pathlib
import numpy as np

path = pathlib.Path.cwd()


def get_data(rinv, N=None):
    df = h5py.File(path.parent / "data" / "jet_images" / f"LL-{rinv}.h5", "r")
    if N is not None:
        X = df["features"][:N]
        y = df["targets"][:N]
    else:
        X = df["features"][:]
        y = df["targets"][:]
    X = np.expand_dims(X, axis=-1)
    X = process_data(X)
    return X, y
