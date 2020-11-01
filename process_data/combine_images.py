import h5py
import numpy as np
import pathlib
import tqdm
path = pathlib.Path.cwd().parent

def combine_images():
    jet_img_path = path / "data" / "jet_images"
    jet_types = ["0p0", "0p3", "1p0", "bkg_qcd"]
    for jet_type in jet_types:
        hf_file = path / "data" / "jet_images" / f"{jet_type}_combined.h5" 
        if not hf_file.exists():
            jet_file = jet_img_path / jet_type
            jet_images = []
            for jet_img in tqdm.tqdm(list(pathlib.Path(jet_file).rglob("*.h5"))):
                jet_img_data = h5py.File(jet_img, "r")["features"][:]
                if len(jet_images) <= 0:
                    jet_images = jet_img_data
                else:
                    jet_images = np.vstack((jet_images, jet_img_data))
                if jet_type == "bkg_qcd" and jet_images.shape[0] > 1300000:
                    break
            if jet_type == "bkg_qcd":
                targets = np.zeros(jet_images.shape[0])
            else:
                targets = np.ones(jet_images.shape[0])

            hf = h5py.File(hf_file, "w")
            hf.create_dataset("features", data=jet_images)
            hf.create_dataset("targets", data=targets)
            hf.close()

if __name__ == "__main__":
    combine_images()
