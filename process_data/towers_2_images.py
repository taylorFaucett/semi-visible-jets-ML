import pandas as pd
import uproot
import numpy as np
import h5py
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import energyflow as ef
from pyjet import cluster
import pickle
from pyjet.testdata import get_event
from scipy.stats import binned_statistic_2d
import pathlib

path = pathlib.Path.cwd().parent


def mass_inv(j1, j2):
    return np.sqrt(
        2.0 * j1.pt * j2.pt * (np.cosh(j1.eta - j2.eta) - np.cos(j1.phi - j2.phi))
    )


def struc2arr(x):
    # pyjet outputs a structured array. This converts
    # the 4 component structured array into a simple
    # 4xN numpy array
    return x.view((float, len(x.dtype.names)))


def jet_trimmer(tower, R0, R1, fcut, pt_min, pt_max, eta_cut):
    # R0 = Clustering radius for the main jets
    # R1 = Clustering radius for the subjets in the primary jet
    trim_pt, trim_eta, trim_phi, trim_mass = [], [], [], []

    # Convert the pandas dataframe to a structured array
    # (pT, eta, phi, mass)
    tower = tower.to_records(index=False)

    # Cluster the event
    sequence = cluster(tower, R=R0, p=-1)
    jets = sequence.inclusive_jets(ptmin=0)

    # check pt and eta cuts
    if pt_min < jets[0].pt < pt_max and -eta_cut < jets[0].eta < +eta_cut:
        # Grab the subjets by clustering with R1
        subjets = cluster(jets[0].constituents_array(), R=R1, p=1)
        subjet_array = subjets.inclusive_jets()

        # For each subjet, check (and trim) based on fcut
        for subjet in subjet_array:
            if subjet.pt > jets[0].pt * fcut:
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

        m_inv = mass_inv(jets[0], jets[1])
        return trimmed_jet, m_inv
    else:
        return None, None


def pixelize(jet):
    # Define the binning for the complete calorimeter
    bins = np.arange(-1.65, 1.65, 0.1)

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
    h5_file_name = (
        path / "data" / "jet_images" / tower_file.parent.stem / f"{tower_file.stem}.h5"
    )
    pkl_trim_name = (
        path
        / "data"
        / "trimmed_jets"
        / tower_file.parent.stem
        / f"{tower_file.stem}.pkl"
    )
    if not pkl_trim_name.parent.exists():
        os.mkdir(pkl_trim_name.parent)
    if not h5_file_name.exists() or not pkl_trim_name.exists():
        tower_events = pd.read_hdf(tower_file, "Tower")
        tower_events = tower_events.astype(np.float64)
        entries = len(tower_events.groupby("entry"))
        jet_images, trimmed_jets, m_invs = [], [], []
        for entry, tower in tower_events.groupby("entry"):
            trimmed_jet, m_inv = jet_trimmer(
                tower=tower,
                R0=1.0,
                R1=0.2,
                fcut=0.05,
                pt_min=300,
                pt_max=400,
                eta_cut=2.0,
            )
            # Pixelize the jet to generate an image
            if trimmed_jet is not None:
                jet_image = pixelize(trimmed_jet)
                jet_images.append(jet_image)
                trimmed_jets.append(trimmed_jet.to_numpy())
                m_invs.append(m_inv)
        jet_images = np.array(jet_images)
        m_invs = np.array(m_invs)
        hf = h5py.File(h5_file_name, "w")
        hf.create_dataset("features", data=jet_images)
        hf.create_dataset("mass", data=m_invs)
        hf.close()
        with open(pkl_trim_name, "wb") as f:
            pickle.dump(trimmed_jets, f)


def towers_2_images():
    # Read tower data exported from root files
    # Convert the tower to a jet-image
    # Save results in data/jet_images
    root_exports_path = path / "data" / "root_exports"
    t = tqdm(list(pathlib.Path(root_exports_path).rglob("**/*.h5")))
    for root_export in t:
        t.set_description(f"Procesing: {root_export.stem}")
        process_tower(root_export)


if __name__ == "__main__":
    towers_2_images()
