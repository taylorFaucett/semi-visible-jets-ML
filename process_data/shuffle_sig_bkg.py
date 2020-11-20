import h5py
import numpy as np
import pathlib
import pickle
import pandas as pd
from sklearn.utils import shuffle


path = pathlib.Path.cwd().parent


def shuffle_sig_bkg():
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        hf_file = path / "data" / "jet_images" / f"LL-{rinv}.h5" 
        mass_file = path / "data" / "jss_observables" / f"mass-{rinv}.h5"
        trim_file = path / "data" / "processed" / f"trimmed-{rinv}.pkl"
        if not hf_file.exists() or not mass_file.exists():
            print("Combining data for file:")
            print(hf_file)
            sig_file = path / "data" / "jet_images" / f"{rinv}_combined.h5"
            sig = h5py.File(sig_file, "r")["features"][:950000]
            sig_mass = h5py.File(sig_file, "r")["mass"][:950000]
            
            sig_length = sig.shape[0]
            jet_images = np.zeros((2*sig_length, 32, 32))
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
            event_count = jet_images.shape[0]
            print(f"Combining Jet Images (N = {event_count})")
            jet_images, targets, masses = shuffle(jet_images, targets, masses, random_state=0)
            
            hf = h5py.File(hf_file, "w")
            hf.create_dataset("features", data=jet_images)
            hf.create_dataset("targets", data=targets)
            hf.close()
            
            hf = h5py.File(mass_file, "w")
            hf.create_dataset("targets", data=targets)
            hf.create_dataset("mass", data=masses)
            hf.close()
            
            jet_images, targets, masses = 0, 0, 0
        
        if not trim_file.exists():
            print(f"Combining Trimmed Events")
            # Trim data
            sig_trim_file = path / "data" / "trimmed_jets" / f"trimmed_jet-{rinv}.pkl"
            bkg_trim_file = path / "data" / "trimmed_jets" / f"trimmed_jet-bkg_qcd.pkl"
                
            print("Loading events")
            sig_trim = pd.read_pickle(sig_trim_file)
            bkg_trim = pd.read_pickle(bkg_trim_file)
            if len(sig_trim) > 950000:
                sig_trim = sig_trim[:950000]
                bkg_trim = bkg_trim[:950000]

            print("Combining")
            sig_trim.extend(bkg_trim)    
            
            print("Shuffling Trimmed Events")
            trim_out = shuffle(sig_trim, random_state=0)
            
            print("Pickling Trimmed Events")
            with open(trim_file, 'wb') as f:
                pickle.dump(trim_out, f)

if __name__ == "__main__":
    shuffle_sig_bkg()
