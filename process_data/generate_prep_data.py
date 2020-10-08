import h5py
import numpy as np
import pandas as pd
import energyflow as ef
import glob
import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import pathlib
path = pathlib.Path.cwd()


def generate_prep_data():
    # Load jet-images
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        jet_img_file = path.parent / "data" / "jet_images" / f"{rinv}.h5"
        jet_images = h5py.File(jet_img_file, mode="r")

        y = jet_images["targets"][:]
        X = jet_images["features"][:]


        eta_ = np.linspace(-0.3875, 0.3875, X.shape[2])
        phi_ = np.linspace(
            -(15.5 * np.pi) / 126.0, (15.5 * np.pi) / 126.0, X.shape[2]
        )


        eta_phi = np.vstack([(x, y) for x in eta_ for y in phi_])
        eta_ = eta_phi[:, 0]
        phi_ = eta_phi[:, 1]
        for ix in trange(len(X)):
            et = X[ix].flatten()
            dfi = pd.DataFrame({"et": et, "eta": eta_, "phi": phi_})
            evt_out = dfi[(dfi[["et"]] != 0).all(axis=1)].to_numpy()
            evt_out[:, 0] /= np.sum(evt_out[:, 0])
            df0.append(evt_out)
        X0 = pd.DataFrame({"features": df0})
        y0 = pd.DataFrame({"targets": y})
        output_path = path.parent / "data" / "processed"
        X0.to_pickle(output_path / rinv / "prep_data.pkl")
        y0.to_pickle(output_path / rinv / "y_prep_data.pkl")


if __name__ == "__main__":
    generate_prep_data()
