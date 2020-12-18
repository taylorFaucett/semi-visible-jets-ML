import h5py
import pathlib
import numpy as np
import pandas as pd
from sklearn import preprocessing

path = pathlib.Path.cwd()


def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x - mean) / std


def get_data(run_type, rinv, N=None):
    if run_type == "LL":
        df = h5py.File(path.parent / "data" / run_type / f"{run_type}-{rinv}.h5", "r")
        X = df["features"][:N]
        y = df["targets"][:N]
        X = np.expand_dims(X, axis=-1)
        X = np.log(1.0 + X) / 4.0
        X = scale_data(X)
    elif run_type == "HL":
        hl_file = path.parent / "data" / run_type / f"HL-{rinv}.h5"
        X = pd.read_hdf(hl_file, "features").to_numpy()[:N].astype("float32")
        y = pd.read_hdf(hl_file, "targets").values[:N]
        scaler = preprocessing.StandardScaler()
        X = scaler.fit_transform(X)
    return X, y
