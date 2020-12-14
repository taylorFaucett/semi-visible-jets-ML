import h5py
import pathlib
import numpy as np
import pandas as pd

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
        if N is not None:
            X = df["features"][:N]
            y = df["targets"][:N]
        else:
            X = df["features"][:]
            y = df["targets"][:]
        X = np.expand_dims(X, axis=-1)
        X = np.log(1.0 + X) / 4.0
    elif run_type == "HL":
        hl_file = path.parent / "data" / run_type / f"HL-{rinv}.h5"
        X = pd.read_hdf(hl_file, "features").to_numpy()
        y = pd.read_hdf(hl_file, "targets").values.astype("int16")
        if N is not None:
            X = X[:N]
            y = y[:N]
    X = scale_data(X)
    return X, y
