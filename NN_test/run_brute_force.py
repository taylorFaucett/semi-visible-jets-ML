# Standard Imports
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import keras
import h5py
from pprint import pprint
import pathlib
import gc
import energyflow as ef
from average_decision_ordering import calc_ado

path = pathlib.Path.cwd()

# Import homemade tools
from get_data import get_data
from get_model import get_model
from mean_ci import mean_ci
from get_best_sherpa_result import get_best_sherpa_result


def select_efps():
    # This function selects a smaller subset of EFPs for the brute force search
    efps_file = path.parent / "data" / "efp" / f"efp-{rinv}.h5"
    efps = list(h5py.File(efps_file, "r")["efps"].keys())

    # Only select d<=5 graphs and prime graphs (i.e. p==1)
    efpset = ef.EFPSet("d<=5", "p==1")

    # Select a subset of kappa beta choices
    kappas = [0.5, 1, 2]
    betas = [0.5, 1, 2]
    graphs = efpset.graphs()
    allowed = []
    for ix, _ in enumerate(graphs):
        for kappa in kappas:
            for beta in betas:
                n, _, d, _, k, _, _, _ = efpset.specs[ix]
                graph_ix = f"{n}_{d}_{k}_k_{kappa}_b_{beta}"
                allowed.append(graph_ix)

    # Some graphs are "duplicates" of existing EFPs
    # Thesea re removed for being put in the EFP h5 file
    # So we can also remove these from the 'allowed' efp list
    # by only keeping common efps between the full list and 'allowed' list
    common_efps = set(efps) - (set(efps) - set(allowed))
    return list(common_efps)


def get_data_w_efp(rinv, chosen_efps, next_efp):
    # Combine HL, previously chosen and next_efp into a training set
    # First get the HL features
    X, y = get_data("HL", rinv)

    # Load the EFP file
    efps_file = path.parent / "data" / "efp" / f"efp-{rinv}.h5"
    efps = h5py.File(efps_file, "r")["efps"]

    # For the previously chosen EFPs, loop through the selections and include to X
    for chosen_efp in chosen_efps:
        chosen_efp = efps[chosen_efp][:].reshape(len(X), 1)
        X = np.hstack((X, chosen_efp))

    # Include the next EFP to check
    next_efp = efps[next_efp][:].reshape(len(X), 1)
    X = np.hstack((X, next_efp))
    return X, y


def run_brute_force(rinv, pass_ix):
    # Setup result and pass data paths and stat files
    bf_path = path / "brute_force_results" / rinv
    pass_path = bf_path / f"pass_{pass_ix}"
    if not bf_path.exists():
        os.mkdir(bf_path)
    if not pass_path.exists():
        os.mkdir(pass_path)

    pass_stats_file = (
        path / "brute_force_results" / rinv / f"pass_{pass_ix}" / "pass_stats.csv"
    )
    if pass_stats_file.exists():
        pass_stats = pd.read_csv(pass_stats_file, index_col=0)
    else:
        pass_stats = pd.DataFrame(columns=["efp", "auc", "ado"])

    # Load sherpa result parameters
    tp = get_best_sherpa_result("HL", rinv)

    # Load CNN predictions to calculate ADO with respect to
    ll_pred = np.load(
        path / "bootstrap_results" / "HL" / rinv / "ll_predictions" / "pred_0.npy"
    )

    # Grab reduced EFP set
    allowed_efps = sorted(select_efps())

    # Remove any EFPs we have already chosen from a previous pass
    for chosen_efp in chosen_efps:
        allowed_efps.remove(chosen_efp)

    # Iterate throgh all the remaining EFPs
    # Train a NN for each and check the AUC/ADO values
    for efp_ix, next_efp in enumerate(allowed_efps):
        X, y = get_data_w_efp(rinv, chosen_efps, next_efp)
        X_train, X_val, y_train, y_val, _, ll_val = train_test_split(
            X, y, ll_pred, test_size=0.75, random_state=42
        )
        input_shape = X_train.shape[1]
        if next_efp not in pass_stats.efp.values:
            print("Running EFP: " + next_efp)
            model = get_model("HL", tp, input_shape)
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    patience=3,
                    min_delta=0.0001,
                    verbose=0,
                    restore_best_weights=True,
                    mode="max",
                )
            ]

            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                verbose=0,
                batch_size=int(tp["batch_size"]),
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )

            # Generate validation predictions and calculate AUC, ADO
            val_predictions = np.hstack(model.predict(X_val))
            auc_val = roc_auc_score(y_val, val_predictions)
            ado_val = calc_ado(val_predictions, np.concatenate(ll_val), y_val, 100000)

            # Store results
            df_ix = pd.DataFrame(
                {"efp": [next_efp], "auc": [auc_val], "ado": [ado_val]}, index=[efp_ix],
            )
            pass_stats = pd.concat([pass_stats, df_ix])
            pass_stats.to_csv(pass_stats_file)
            print(f"    EFP: {next_efp} -> AUC = {auc_val:0.4f}, ADO = {ado_val:0.4f}")

    # Isolate the best performning EFP by ADO value and return that result to be included in 'chosen_efps'
    sorted_stats = pass_stats.sort_values(by="ado", ascending=False)
    best_result = sorted_stats.iloc[0]
    return best_result


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        epochs = 100

        # Number of brute force attempts
        N_passes = 5

        # Best choices from each pass are collected in these lists
        chosen_efps, aucs, ados = [], [], []

        # Start with a first pass (i.e. start just with HL)
        pass_ix = 0
        while pass_ix <= N_passes:
            # Run Brute Force Search
            pass_stats = run_brute_force(rinv, pass_ix)
            pass_ix += 1

            # Collect results and save them
            chosen_efps.append(pass_stats.efp)
            aucs.append(pass_stats.auc)
            ados.append(pass_stats.ado)
            final_results = pd.DataFrame({"efp": chosen_efps, "auc": aucs, "ado": ados})
            final_results.to_csv(
                path / "brute_force_results" / rinv / "brute_force_results.csv"
            )
