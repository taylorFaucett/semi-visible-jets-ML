import h5py
import pathlib
import numpy as np
import pandas as pd
from sklearn import preprocessing

path = pathlib.Path.cwd()


class getData:
    def __init__(self, run_type, rinv, N):
        self.run_type = run_type
        self.rinv = rinv
        self.N = N

    def scaleData(self, x, mean=True):
        mean = np.mean(x)
        std = np.std(x)
        if std == 0:
            std = 1
        return (x - mean) / std

    def features(self):
        if self.run_type == "LL":
            df = h5py.File(path.parent / "data" / self.run_type / f"{self.run_type}-{self.rinv}.h5", "r")
            X = df["features"][:self.N]
            X = np.expand_dims(X, axis=-1)
            X = np.log(1.0 + X) / 4.0
            X = self.scaleData(X)
        elif self.run_type == "HL":
            hl_file = path.parent / "data" / self.run_type / f"HL-{self.rinv}.h5"
            X = pd.read_hdf(hl_file, "features").to_numpy()[:self.N].astype("float32")
            scaler = preprocessing.StandardScaler()
            X = scaler.fit_transform(X)
        return X
    
    def targets(self):
        hf = h5py.File(path.parent / "data" / "LL" / f"LL-{self.rinv}.h5", "r")
        y = hf["targets"][:self.N]
        return y
