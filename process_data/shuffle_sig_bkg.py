import h5py
import numpy as np
import pathlib
import random

path = pathlib.Path.cwd().parent


def shuffle_sig_bkg():
    rinvs = ["0p0", "0p3", "1p0"]
    max_size = 1100000
    for rinv in rinvs:
        hf_file = path / "data" / "jet_images" / f"LL-{rinv}.h5" 
        if not hf_file.exists():
            print("Combining data for file:")
            print(hf_file)
            sig_file = path / "data" / "jet_images" / f"{rinv}_combined.h5"
            sig = h5py.File(sig_file, "r")["features"][:]
            sig_y = h5py.File(sig_file, "r")["targets"][:]
            sig_length = len(sig_y)
            jet_images = np.zeros((2*sig_length, 32, 32))
            jet_images[:sig.shape[0],:] = sig
            sig = 1
            
            bkg_file = path / "data" / "jet_images" / "bkg_qcd_combined.h5"
            bkg = h5py.File(bkg_file, "r")["features"][:sig_length]
            bkg_y = h5py.File(bkg_file, "r")["targets"][:sig_length]
            jet_images[bkg.shape[0]:,:] = bkg
            bkg=0
            
            #jet_images = np.append(sig, bkg, axis=0)
            targets = np.hstack((sig_y, bkg_y))
            np.random.seed(0)
            np.random.shuffle(jet_images)
            np.random.shuffle(targets)
            hf = h5py.File(hf_file, "w")
            hf.create_dataset("features", data=jet_images)
            hf.create_dataset("targets", data=targets)
            hf.close()

if __name__ == "__main__":
    shuffle_sig_bkg()
