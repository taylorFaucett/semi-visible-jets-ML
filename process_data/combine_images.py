import h5py
import numpy as np
import pathlib
path = pathlib.Path.cwd().parent

def combine_images():
    jet_img_path = path / "data" / "jet_images"
    jet_types = ["0p0", "0p3", "1p0", "bkg_qcd"]
    for jet_type in jet_types:
        jet_file = jet_img_path / jet_type
        jet_images = []
        for jet_img in pathlib.Path(jet_file).rglob("*.h5"):
            print(jet_img)
            jet_img_data = h5py.File(jet_img, "r")["features"][:]
            if len(jet_images) <= 0:
                jet_images = jet_img_data
            else:
                jet_images = np.vstack((jet_images, jet_img_data))
        if jet_type == "bkg_qcd":
            targets = np.zeros(len(jet_images))
        else:
            targets = np.ones(len(jet_images))
        hf_file = path / "data" / "jet_images" / f"{jet_type}_combined.h5" 
        hf = h5py.File(hf_file, "w")
        hf.create_dataset("features", data=jet_images)
        hf.create_dataset("targets", data=targets)
        hf.close()

if __name__ == "__main__":
    combine_images()