# Standard Imports
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
import h5py
import keras
import pathlib
import gc

path = pathlib.Path.cwd()

# Import homemade tools
from get_data import get_data
from get_model import get_model


def generator(data_type, rinv, batch_size):
    while True:
        f = h5py.File(path / "temp_h5" / f"{data_type}_temp.h5", "r")
        N = f["targets"].shape[0]
        start = 0
        end = batch_size
        while start < N:
            X = f["features"][start:end]
            y = f["targets"][start:end]
            yield X, y
            start += batch_size
            end += batch_size


def gen_temp_h5(data_type, X, y):
    hf = h5py.File(path / "temp_h5" / f"{data_type}_temp.h5", "w")
    hf.create_dataset("features", data=X)
    hf.create_dataset("targets", data=y)
    hf.close()


def run_bootstraps(rinv):
    # Trainig parameters from the sherpa optimization
    tp = np.load(f"sherpa_results/{rinv}.npy", allow_pickle="TRUE").item()
    X, y = get_data(rinv=rinv)
    n = len(y)
    rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.10)
    rs.get_n_splits(X)
    ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.10)
    straps = []
    aucs = []
    bs_count = 0
    t = tqdm(list(rs.split(X)))
    counter = 0

    for train_index, test_index in t:
        X_val = X[test_index]
        y_val = y[test_index]
        N_size = len(y[train_index])

        # Save a temp h5 for the generator
        gen_temp_h5("training", X[train_index], y[train_index])
        X, y = 0, 0

        bs_path = path / "bootstrap_results" / rinv
        if not bs_path.exists():
            os.mkdir(bs_path)
        model_file = bs_path / "models" / f"bs_{bs_count}.h5"
        roc_file = bs_path / "roc" / f"roc_{bs_count}.csv"
        if not model_file.parent.exists():
            os.mkdir(model_file.parent)

        if not roc_file.parent.exists():
            os.mkdir(roc_file.parent)

        if not model_file.exists():
            model = get_model(tp=tp)
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
            batch_size = int(tp["batch_size"])
            training_data = generator(
                data_type="training", rinv=rinv, batch_size=batch_size
            )
            history = model.fit(
                training_data,
                epochs=epochs,
                verbose=2,
                steps_per_epoch=int(N_size * 0.80) // batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
            )

        else:
            model = keras.models.load_model(model_file)

        val_predictions = np.hstack(model.predict(X_val))
        auc_val = roc_auc_score(y_val, val_predictions)
        straps.append(bs_count)
        aucs.append(auc_val)

        fpr, tpr, thresholds = roc_curve(y_val, val_predictions)
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
        auc_mean = np.average(aucs)
        auc_ci = np.percentile(aucs, (2.5, 97.5))
        auc_ci_max = max(np.abs(auc_ci[0] - auc_mean), np.abs(auc_ci[1] - auc_mean))
        bs_count += 1
        counter += 1
        t.set_description(
            f"rinv={rinv} ({counter}/{n_splits}): (AUC = {auc_mean:.4f} +/- {auc_ci_max:.4f}"
        )
        t.refresh()
        gc.collect()


if __name__ == "__main__":
    rinvs = ["0p3", "0p0", "1p0"]
    n_splits = 2
    epochs = 200
    for rinv in rinvs:
        run_bootstraps(rinv)
