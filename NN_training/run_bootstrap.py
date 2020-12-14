# Standard Imports
import os
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import keras
import pathlib
import gc

path = pathlib.Path.cwd()

# Import homemade tools
from get_data import get_data
from get_model import get_model
from mean_ci import mean_ci


def get_best_sherpa_result(run_type, rinv):
    sherpa_file = path / "sherpa_results" / run_type / rinv / "results.csv"
    sherpa_results = pd.read_csv(sherpa_file, index_col="Trial-ID").groupby("Status")
    sorted_results = sherpa_results.get_group("COMPLETED").sort_values(
        by="Objective", ascending=False
    )
    best_result = sorted_results.iloc[0].to_dict()
    return best_result


def run_bootstraps(run_type, rinv):
    # Trainig parameters from the sherpa optimization
    tp = get_best_sherpa_result(run_type, rinv)
    X, y = get_data(run_type, rinv)

    rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.10)
    rs.get_n_splits(X)
    ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.10)
    straps = []
    aucs = []
    boot_ix = 0
    t = tqdm(list(rs.split(X)))

    for train_index, test_index in t:
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[test_index]
        y_val = y[test_index]

        if run_type == "HL":
            input_shape = X_train.shape[1]
        else:
            input_shape = None

        bs_path = path / "bootstrap_results" / run_type / rinv
        model_file = bs_path / "models" / f"bs_{boot_ix}.h5"
        roc_file = bs_path / "roc" / f"roc_{boot_ix}.csv"
        ll_pred_file = bs_path / "ll_predictions" / f"pred_{boot_ix}.npy"

        if not bs_path.parent.exists():
            os.mkdir(bs_path.parent)
         if not bs_path.exists():   
            os.mkdir(bs_path)
        if not model_file.parent.exists():
            os.mkdir(model_file.parent)
        if not roc_file.parent.exists():
            os.mkdir(roc_file.parent)
        if not ll_pred_file.parent.exists():
            os.mkdir(ll_pred_file.parent)

        if not model_file.exists():
            model = get_model(run_type, tp, input_shape)
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_auc",
                    patience=10,
                    min_delta=0.0001,
                    verbose=0,
                    restore_best_weights=True,
                    mode="max",
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=model_file, verbose=0, save_best_only=True
                ),
            ]

            model.fit(
                X_train,
                y_train,
                epochs=epochs,
                verbose=2,
                batch_size=int(tp["batch_size"]),
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )

        else:
            model = keras.models.load_model(model_file)

        val_predictions = np.hstack(model.predict(X_val))
        auc_val = roc_auc_score(y_val, val_predictions)

        # Save the predictions
        np.save(ll_pred_file, model.predict(X))

        straps.append(boot_ix)
        aucs.append(auc_val)

        fpr, tpr, _ = roc_curve(y_val, val_predictions)
        background_efficiency = fpr
        signal_efficiency = tpr
        background_rejection = 1.0 - background_efficiency

        roc_df = pd.DataFrame(
            {
                "sig_eff": signal_efficiency,
                "bkg_eff": background_efficiency,
                "bkg_rej": background_rejection,
            }
        )
        roc_df.to_csv(roc_file)

        results = pd.DataFrame({"bs": straps, "auc": aucs})
        results.to_csv(bs_path / "aucs.csv")
        auc_mean, auc_ci = mean_ci(aucs)
        boot_ix += 1
        t.set_description(
            f"rinv={rinv} ({boot_ix}/{n_splits}): (AUC = {auc_mean:.4f} +/- {auc_ci:.4f})"
        )
        t.refresh()


if __name__ == "__main__":
    run_type = str(sys.argv[1])
    rinvs = ["0p0", "0p3", "1p0"]
    n_splits = 5
    epochs = 200
    for rinv in rinvs:
        run_bootstraps(run_type, rinv)
