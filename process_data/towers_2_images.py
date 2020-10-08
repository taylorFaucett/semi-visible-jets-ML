import pandas as pd
import uproot
import numpy as np
import h5py
import glob
import matplotlib.pyplot as plt
import os
from pyjet import cluster
from pyjet.testdata import get_event
from scipy.stats import binned_statistic_2d
import pathlib
path = pathlib.Path.cwd().parent


def struc2arr(x):
    # pyjet outputs a structured array. This converts
    # the 4 component structured array into a simple
    # 4xN numpy array
    x = x.view((float, len(x.dtype.names)))
    return x


def jet_trimmer(tower, R0, R1, ptmin):
    # R0 = Clustering radius for the main jets
    # R1 = Clustering radius for the subjets in the primary jet
    trim_pt, trim_eta, trim_phi, trim_mass = [], [], [], []

    # Convert the pandas dataframe to a structured array
    # (pT, eta, phi, mass)
    tower = tower.to_records(index=False)

    # Cluster the event
    sequence = cluster(tower, R=R0, p=-1)
    jets = sequence.inclusive_jets(ptmin=ptmin)

    # Take just the leading jet 
    # If there isn't one, drop out with an empty event
    try:
        jet0 = jets[0]
    except:
        z1 = np.zeros(1)
        return pd.DataFrame({"pt":z1 , "eta": z1, "phi": z1, "mass": z1})
        

    # Define a cut threshold that the subjets have to meet (i.e. 5% of the original jet pT)
    jet0_max = jet0.pt
    jet0_cut = jet0_max * 0.05

    # Grab the subjets by clustering with R1
    subjets = cluster(jet0.constituents_array(), R=R1, p=1)
    subjet_array = subjets.inclusive_jets()
    for subjet in subjet_array:
        if subjet.pt > jet0_cut:
            # Get the subjets pt, eta, phi constituents
            subjet_data = subjet.constituents_array()
            subjet_data = struc2arr(subjet_data)
            pT = subjet_data[:, 0]
            eta = subjet_data[:, 1]
            phi = subjet_data[:, 2]
            mass = subjet_data[:, 3]

            # Shift all data such that the leading subjet
            # is located at (eta,phi) = (0,0)
            eta -= subjet_array[0].eta
            phi -= subjet_array[0].phi

            # Collect the trimmed subjet constituents
            trim_pt.extend(pT)
            trim_eta.extend(eta)
            trim_phi.extend(phi)
            trim_mass.extend(mass)

    trimmed_jet = pd.DataFrame(
        {"pt": trim_pt, "eta": trim_eta, "phi": trim_phi, "mass": trim_mass}
    )
    
    return trimmed_jet


def pixelize(jet):
    # Define the binning for the complete calorimeter
    bins = np.arange(-1.6, 1.6, 0.1)

    # Sum energy deposits in each bin
    digitized = binned_statistic_2d(
        jet.eta, jet.phi, jet.pt, statistic="sum", bins=bins
    )
    jet_image = digitized.statistic
    return jet_image


def process_tower(tower_file):
    h5_dir = path / "data" / "jet_images" / tower_file.parent.stem
    if not os.path.exists(h5_dir):
        os.mkdir(h5_dir)
    h5_file_name = path / "data" / "jet_images" / tower_file.parent.stem / f"{tower_file.stem}.h5"
    if not h5_file_name.exists():
        tower_events = pd.read_hdf(tower_file, "Tower")
        tower_events = tower_events.astype(np.float64)
        entries = len(tower_events.groupby("entry"))
        jet_images = np.zeros((entries, 31, 31))
        for entry, tower in tower_events.groupby("entry"):
            trimmed_jet = jet_trimmer(tower, 1.0, 0.3, 0.05)
            # Pixelize the jet to generate an image
            jet_image = pixelize(trimmed_jet)
            jet_images[entry] = jet_image
        hf = h5py.File(h5_file_name, "w")
        hf.create_dataset("features", data=jet_images)
        hf.close()

def towers_2_images():
    # Read tower data exported from root files
    # Convert the tower to a jet-image
    # Save results in data/jet_images
    root_exports_path = path / "data" / "root_exports"
    for root_export in pathlib.Path(root_exports_path).rglob("**/*.h5"):
        print(f"working on: {root_export}")
        try:
            process_tower(root_export)
        except:
            pass

if __name__ == "__main__":
    towers_2_images()
    
    
    
