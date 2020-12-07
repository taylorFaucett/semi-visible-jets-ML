import uproot
import numpy as np
import glob
import pandas as pd
import os
from tqdm import tqdm

home = os.getcwd()


def export_data(root_file, branches):
    events = uproot.open(root_file)["Delphes"]
    pt_eta_phi_mass = events.arrays(
        branches=branches, outputtype=pd.DataFrame, flatten=True
    )
    return pt_eta_phi_mass


def root_df():
    rinvs = ["0p0", "0p3", "1p0"]
    branch_list = [
        ["Jet.PT", "Jet.Eta", "Jet.Phi", "Jet.Mass"],
        ["Tower.ET", "Tower.Eta", "Tower.Phi", "Tower.E"],
    ]

    for rinv in rinvs:
        rinv_path = os.path.join(home, "root_exports", rinv)
        if not os.path.exists(rinv_path):
            os.mkdir(rinv_path)

        root_files = glob.glob(
            os.path.join(home, "data", "root_files", "rinv-" + rinv, "*.root")
        )
        for root_file in tqdm(root_files):
            run_name = os.path.basename(root_file).split(".root")[0]
            output_file = os.path.join(home, "root_exports", rinv, run_name + ".h5")
            if not os.path.isfile(output_file):
                for branches in branch_list:
                    pt_eta_phi_mass = export_data(root_file, branches)
                    key = branches[0].split(".")[0]
                    pt_eta_phi_mass.to_hdf(output_file, key=key, mode="a")


if __name__ == "__main__":
    root_df()
