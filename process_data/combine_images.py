import h5py
import numpy as np
import pathlib
import tqdm
import pandas as pd
import pickle

path = pathlib.Path.cwd().parent


def combine_images():
    jet_img_path = path / "data" / "jet_images"
    jet_types = ["0p0", "0p3", "1p0", "bkg_qcd"]
    for jet_type in jet_types:
        hf_file = path / "data" / "jet_images" / f"{jet_type}_combined.h5"
        if not hf_file.exists():
            jet_file = jet_img_path / jet_type

            jet_images, masses, trim_jets = [], [], []
            for jet_img in tqdm.tqdm(list(pathlib.Path(jet_file).rglob("*.h5"))):
                jet_img_data = h5py.File(jet_img, "r")["features"][:]
                mass_data = h5py.File(jet_img, "r")["mass"][:]
                trim_jet = pd.read_pickle(
                    hf_file.parent.parent
                    / "trimmed_jets"
                    / jet_type
                    / f"{jet_img.stem}.pkl"
                )
                if len(jet_images) <= 0:
                    jet_images = jet_img_data
                    masses = mass_data
                else:
                    jet_images = np.vstack((jet_images, jet_img_data))
                    masses = np.hstack((masses, mass_data))
                trim_jets.extend(trim_jet)
                if jet_type == "bkg_qcd" and jet_images.shape[0] > 1300000:
                    # This is only here because we have a lot more bkg data (largest sig is 1.2 Million)
                    # So I stop the file getting bigger than we need for any of our signal options
                    break
            if jet_type == "bkg_qcd":
                targets = np.zeros(len(masses))
            else:
                targets = np.ones(len(masses))

            hf = h5py.File(hf_file, "w")
            hf.create_dataset("features", data=jet_images)
            hf.create_dataset("targets", data=targets)
            hf.create_dataset("mass", data=masses)
            hf.close()

            with open(
                hf_file.parent.parent / "trimmed_jets" / f"trimmed_jet-{jet_type}.pkl",
                "wb",
            ) as f:
                pickle.dump(trim_jets, f)


if __name__ == "__main__":
    combine_images()
