import h5py
from sklearn import preprocessing
from process_data import process_data
import pathlib

path = pathlib.Path.cwd()


def get_data(rinv, N=None):
    df = h5py.File(path.parent / "data" / "jet_images" / f"LL-{rinv}.h5", "r")
    X = df["features"][:].astype("float32")
    y = df["targets"][:]
    if N is not None:
        X = X[: N - 1]
        y = y[: N - 1]
    X = process_data(X)
    return X, y
