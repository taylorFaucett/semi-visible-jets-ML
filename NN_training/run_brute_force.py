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
import pathlib
import gc
import energyflow as ef
from average_decision_ordering import calc_ado

path = pathlib.Path.cwd()

# Import homemade tools
from get_data import get_data
from get_model import get_model
from mean_ci import mean_ci


def get_best_sherpa_result(rinv):
    sherpa_file = path / "sherpa_results" / "HL" / rinv / "results.csv"
    sherpa_results = pd.read_csv(sherpa_file, index_col="Trial-ID").groupby("Status")
    sorted_results = sherpa_results.get_group("COMPLETED").sort_values(
        by="Objective", ascending=False
    )
    best_result = sorted_results.iloc[0].to_dict()
    return best_result


def select_efps():
    efps_file = path.parent / "data" / "efp" / f"efp-{rinv}.h5"
    efps = list(h5py.File(efps_file, "r")["efps"].keys())
    efpset = ef.EFPSet("d<=5", "p==1")
    kappas = [0.5, 1, 2]
    betas = [0.5, 1, 2]
    graphs = efpset.graphs()
    allowed = []
    for ix, graph in enumerate(graphs):
        for kappa in kappas:
            for beta in betas:
                n, e, d, v, k, c, p, h = efpset.specs[ix]
                graph_ix = f"{n}_{d}_{k}_k_{kappa}_b_{beta}"
                allowed.append(graph_ix)
    common_efps = set(efps) - (set(efps) - set(allowed))
    return list(common_efps)


def get_data_w_efp(rinv, chosen_efps, next_efp):
    X, y = get_data("HL", rinv)
    efps_file = path.parent / "data" / "efp" / f"efp-{rinv}.h5"
    efps = h5py.File(efps_file, "r")["efps"]
    for chosen_efp in chosen_efps:
        chosen_efp = efps[chosen_efp][:].reshape(len(X), 1)
        X = np.hstack((X, chosen_efp))
    next_efp = efps[next_efp][:].reshape(len(X), 1)
    X = np.hstack((X, next_efp))
    return X, y


def run_brute_force(rinv, pass_ix):
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

    tp = get_best_sherpa_result(rinv)
    ll_pred = np.load(
        path / "bootstrap_results" / "HL" / rinv / "ll_predictions" / "pred_0.npy"
    )

    allowed_efps = select_efps()
    for chosen_efp in chosen_efps:
        allowed_efps.remove(chosen_efp)

    for efp_ix, next_efp in enumerate(tqdm(allowed_efps)):
        X, y = get_data_w_efp(rinv, chosen_efps, next_efp)
        if X is not None:
            X_train, X_val, y_train, y_val, ll_train, ll_val = train_test_split(
                X, y, ll_pred, test_size=0.75, random_state=42
            )
            input_shape = X_train.shape[1]
            if next_efp not in pass_stats.efp:
                model = get_model("HL", tp, input_shape)
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_auc",
                        patience=10,
                        min_delta=0.0001,
                        verbose=0,
                        restore_best_weights=True,
                        mode="max",
                    )
                ]

                history = model.fit(
                    X_train,
                    y_train,
                    epochs=epochs,
                    verbose=0,
                    batch_size=int(tp["batch_size"]),
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                )

                val_predictions = np.hstack(model.predict(X_val))
                auc_val = roc_auc_score(y_val, val_predictions)
                ado_val = calc_ado(
                    val_predictions, np.concatenate(ll_val), y_val, 100000
                )
                df_ix = pd.DataFrame(
                    {"efp": [next_efp], "auc": [auc_val], "ado": [ado_val]},
                    index=[efp_ix],
                )
                pass_stats = pd.concat([pass_stats, df_ix])
                pass_stats.to_csv(pass_stats_file)
    sorted_stats = pass_stats.sort_values(by="ado", ascending=False)
    best_result = sorted_stats.iloc[0]
    return best_result


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        epochs = 50
        chosen_efps, aucs, ados = [], [], []
        pass_ix = 0
        while pass_ix <= 5:
            pass_stats = run_brute_force(rinv, pass_ix)
            pass_ix += 1
            chosen_efps.append(pass_stats.efp)
            aucs.append(pass_stats.auc)
            ados.append(pass_stats.ado)
            final_results = pd.DataFrame({"efp": chosen_efps, "auc": aucs, "ado": ados})
            final_results.to_csv(path / "brute_force_results" / "results.csv")
