import pandas as pd
from sklearn import preprocessing
import pathlib

path = pathlib.Path.cwd()


def get_data(rinv, N=None):
    hl_file = path.parent / "data" / "jss_observables" / f"HL-{rinv}.h5"
    X = pd.read_hdf(hl_file, "features")
    y = pd.read_hdf(hl_file, "targets")

    if N is not None:
        X = X.loc[:N]
        y = y.loc[:N]
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    return X, y
