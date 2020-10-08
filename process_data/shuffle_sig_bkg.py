import h5py
import numpy as np
import pathlib
import random

path = pathlib.Path.cwd().parent


def shuffle_sig_bkg():
    bkg_file = path / "data" / "jet_images" / "bkg_qcd_combined.h5"
    bkg = h5py.File(bkg_file, "r")["features"][:]
    bkg_y = h5py.File(bkg_file, "r")["targets"][:]
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        sig_file = path / "data" / "jet_images" / f"{rinv}_combined.h5"
        sig = h5py.File(bkg_file, "r")["features"][:]
        sig_y = h5py.File(bkg_file, "r")["targets"][:]
        jet_images = np.vstack((sig, bkg))
        targets = np.vstack((sig_y, bkg_y))
        np.random.seed(0)
        jet_images = np.random.shuffle(jet_images)
        targets = np.random.shuffle(targets)
        
        hf_file = path / "data" / "jet_images" / f"LL-{rinv}.h5" 
        hf = h5py.File(hf_file, "w")
        hf.create_dataset("features", data=jet_images)
        hf.create_dataset("targets", data=targets)
        hf.close()

if __name__ == "__main__":
    shuffle_sig_bkg()