import h5py
import numpy as np
import pathlib
from sklearn.utils import shuffle


path = pathlib.Path.cwd().parent


def shuffle_sig_bkg():
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        hf_file = path / "data" / "jet_images" / f"LL-{rinv}.h5" 
        mass_file = path / "data" / "jss_observables" / f"mass-{rinv}.h5"
        if not hf_file.exists():
            print("Combining data for file:")
            print(hf_file)
            sig_file = path / "data" / "jet_images" / f"{rinv}_combined.h5"
            sig = h5py.File(sig_file, "r")["features"][:950000]
            sig_mass = h5py.File(sig_file, "r")["mass"][:950000]
            
            sig_length = sig.shape[0]
            jet_images = np.zeros((2*sig_length, 32, 32))
            print(jet_images.shape)
            jet_images[:sig.shape[0],:] = sig
            sig = 1
            bkg_file = path / "data" / "jet_images" / "bkg_qcd_combined.h5"
            bkg = h5py.File(bkg_file, "r")["features"][:sig_length]
            bkg_mass = h5py.File(bkg_file, "r")["mass"][:sig_length]
            jet_images[bkg.shape[0]:,:] = bkg
            bkg=0
            
            #jet_images = np.append(sig, bkg, axis=0)
            sig_y = h5py.File(sig_file, "r")["targets"][:sig_length]
            bkg_y = h5py.File(bkg_file, "r")["targets"][:sig_length]
            targets = np.hstack((sig_y, bkg_y))
            masses = np.hstack((sig_mass, bkg_mass))
            
            jet_images, targets, masses = shuffle(jet_images, targets, masses, random_state=0)
            hf = h5py.File(hf_file, "w")
            hf.create_dataset("features", data=jet_images)
            hf.create_dataset("targets", data=targets)
            hf.close()
            
            hf = h5py.File(mass_file, "w")
            hf.create_dataset("targets", data=targets)
            hf.create_dataset("mass", data=masses)
            hf.close()

if __name__ == "__main__":
    shuffle_sig_bkg()
