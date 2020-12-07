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
        hdf_out = path.parent / "data" / "processed" / f"{rinv}-prep_data.h5"
        pT_out = path.parent / "data" / "jss_observables" / f"pT-{rinv}.feather"
        if not hdf_out.exists():
            jet_img_file = path.parent / "data" / "jet_images" / f"LL-{rinv}.h5"
            jet_images = h5py.File(jet_img_file, mode="r")
            y = jet_images["targets"][:]
            X = jet_images["features"][:]

            eta_ = np.linspace(-0.4, 0.4, X.shape[2])
            phi_ = np.linspace(-(4 * np.pi) / 31.0, (4 * np.pi) / 31.0, X.shape[2])

            eta_phi = np.vstack([(x, y) for x in eta_ for y in phi_])
            eta_ = eta_phi[:, 0]
            phi_ = eta_phi[:, 1]
            df0 = []
            raw_pT = []
            for ix in trange(X.shape[0]):
                et = X[ix].flatten()
                dfi = pd.DataFrame({"et": et, "eta": eta_, "phi": phi_})
                evt_out = dfi[(dfi[["et"]] != 0).all(axis=1)].to_numpy()
                raw_pT.append(np.sum(dfi.et.values))
                evt_out[:, 0] /= np.sum(evt_out[:, 0])
                df0.append(evt_out)
            X0 = pd.DataFrame({"features": df0})
            y0 = pd.DataFrame({"targets": y})
            X0.to_hdf(hdf_out, "features", mode="a")
            y0.to_hdf(hdf_out, "targets", mode="a")

            pT_df = pd.DataFrame({"pT": raw_pT})
            pT_df.to_feather(pT_out)


if __name__ == "__main__":
    generate_prep_data()
