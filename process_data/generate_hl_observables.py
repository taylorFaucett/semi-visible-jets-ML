import os
import pandas as pd
import numpy as np
import pathlib
import h5py
import glob
np.warnings.filterwarnings('ignore')

path = pathlib.Path.cwd()

for entry in os.scandir('JSS'):
    if entry.is_file() and "init" not in str(entry) and "jss_template" not in str(entry):
        string = f'from JSS import {entry.name}'[:-3]
        exec(string)

def load_modules():
    # Anything placed in JSS will be imported and used as a HL observable.
    # The input will be the prep_data file
    # The output should be a simple 1D numpy array
    jss_list = glob.glob("JSS/*.py")
    jss_list = [x.split("/")[-1].split(".py")[0] for x in jss_list]
    jss_list.remove("__init__")
    jss_list.remove("jss_template")
    return jss_list

def generate_hl_observables():
    quick_run=False
    JSS_list = load_modules()
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        h5_file = path.parent / "data" / "jss_observables" / f"HL-{rinv}.h5"
        if h5_file.exists():
            HL_df = pd.read_hdf(h5_file, "features")
            existing_jss = list(HL_df.columns)
        else:
            HL_df = pd.DataFrame()
            jet_mass = pd.DataFrame({"mass": h5py.File(path.parent / "data" / "jss_observables" / f"mass-{rinv}.h5", "r")["mass"][:]})
            jet_pt = pd.read_feather(path.parent / "data" / "jss_observables" / f"pT-{rinv}.feather")
            HL_df = pd.concat([jet_mass, jet_pt], axis=1)
            existing_jss = []
        prep_data = path.parent / "data" / "processed" / f"{rinv}-prep_data.h5"
        trim_data = path.parent / "data" / "processed" / f"trimmed-{rinv}.pkl"
        
        if quick_run:
            X = pd.read_hdf(prep_data, "features").features.to_numpy()[:2000]
            #X_trim = pd.read_pickle(trim_data)[:20000]
            y = pd.read_hdf(prep_data, "targets").targets[:2000]
        else:
            X = pd.read_hdf(prep_data, "features").features.to_numpy()
            #X_trim = pd.read_pickle(trim_data)
            y = pd.read_hdf(prep_data, "targets").targets        
        for JSS_calc in JSS_list:
            if JSS_calc not in existing_jss:
                print(f"Calculating {JSS_calc} on data set: {rinv}")
                try:
                    JSS_out = np.zeros(X.shape[0])
                    exec("JSS_out[:] = %s.calc(X)[:]" %JSS_calc)
                    JSS_out = pd.DataFrame({JSS_calc:JSS_out})
                    HL_df = pd.concat([HL_df, JSS_out], axis=1)
                except Exception as e:
                    print(f"JSS calculation for {JSS_calc} on data set {rinv} failed with error:")
                    print(e)

            # Re-organize columns alphabettically. 
            # This guarantees the ordering is always the same
            HL_df = HL_df.reindex(sorted(HL_df.columns), axis=1)
            print(HL_df)
            HL_df.to_hdf(h5_file , key="features", mode="w")
            y.to_hdf(h5_file , key="targets", mode="a")
        

if __name__ == "__main__":
    generate_hl_observables()