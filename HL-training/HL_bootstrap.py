# Standard Imports
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import keras
import pathlib

path = pathlib.Path.cwd()

# Import homemade tools
from get_data import get_data
from HL_model import HL_model


def run_bootstraps(rinv):
    # Trainig parameters from the sherpa optimization
    tp = np.load(f"sherpa_results/{rinv}.npy", allow_pickle="TRUE").item()
    X, y = get_data(rinv)

    n = len(y)
    n_train = int(0.85 * n)
    n_test = int(0.15 * n)
    rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.15)
    rs.get_n_splits(X)
    ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.15)
    straps = []
    aucs = []
    bs_count = 0
    t = tqdm(list(rs.split(X)))
    counter = 0

    for train_index, test_index in t:
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[test_index]
        y_val = y[test_index]
        model_file = bs_path / "models" / f"bs-{bs_count}.h5"
        if not model_file.parent.exists():
            os.mkdir(model_file.parent)

        if not model_file.exists():
            model = HL_model(tp=tp, input_shape=X_train.shape[1])
            callbacks = [
                keras.callbacks.EarlyStopping(
                    patience=10, verbose=0, restore_best_weights=True
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath=model_file, verbose=0, save_best_only=True
                ),
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

        else:
            model = keras.models.load_model(model_file)

        val_predictions = np.hstack(model.predict(X_val))
        auc_val = roc_auc_score(y_val, val_predictions)
        straps.append(bs_count)
        aucs.append(auc_val)

        results = pd.DataFrame({"bs": straps, "auc": aucs})
        results.to_csv(bs_path / "bootstrap_results.csv")
        auc_mean = np.average(aucs)
        auc_std = np.std(aucs)
        bs_count += 1
        counter += 1
        t.set_description(
            f"Run ({counter}/{n_splits}): (AUC = {auc_mean:.5f} +/- {auc_std:.5f}"
        )
        t.refresh()


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    n_splits = 5
    epochs = 200
    for rinv in rinvs:
        bs_path = path / "bootstrap_results" / rinv
        if not bs_path.exists():
            os.mkdir(bs_path)
        run_bootstraps(rinv)
