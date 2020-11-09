import os
import pandas as pd
import numpy as np
import pathlib
import glob

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
    JSS_list = load_modules()
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        HL_df = pd.DataFrame()
        for JSS_calc in JSS_list:
            hdf_file = path.parent / "data" / "processed" / f"{rinv}-prep_data.h5"
            print(f"Calculating {JSS_calc} on data set: {rinv}")
            X = pd.read_hdf(hdf_file, "features").features.to_numpy()
            y = pd.read_hdf(hdf_file, "targets").targets
            try:
                JSS_out = np.zeros(X.shape[0])
                exec("JSS_out[:] = %s.calc(X)[:]" %JSS_calc)
                JSS_out = pd.DataFrame({JSS_calc:JSS_out})
                HL_df = pd.concat([HL_df, JSS_out], axis=1)
            except Exception as e:
                print(f"JSS calculation for {JSS_calc} on data set {rinv} failed with error:")
                print(e)
                exit()
        # HL_df = pd.concat([HL_df, pd.DataFrame({"targets":y})], axis=1)
        h5_file = path.parent / "data" / "jss_observables" / f"HL-{rinv}.h5"
        HL_df.to_hdf(h5_file , key="features", mode="w")
        y.to_hdf(h5_file , key="targets", mode="a")
        
    
    

if __name__ == "__main__":
    generate_hl_observables()