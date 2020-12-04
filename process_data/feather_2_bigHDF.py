import pandas as pd
import h5py
import tqdm
import glob
import pathlib
path = pathlib.Path.cwd()

rinvs = ["0p0", "0p3", "1p0"]
for rinv in rinvs:
    hf = h5py.File(path.parent / "data" / "efp" / f"efp-{rinv}.h5", "w")
    targets = pd.read_feather(path.parent / "data" / "efp" / rinv / "1_0_0_k_-1_b_0.5.feather").targets.values
    hf.create_dataset("targets",data=targets)
    g1 = hf.create_group("efps")
    for efp_file in tqdm.tqdm(list(pathlib.Path(path.parent / "data" / "efp" / rinv).rglob("*"))):
        efp_data = pd.read_feather(efp_file).features.values
        g1.create_dataset(efp_file.stem,data=efp_data)