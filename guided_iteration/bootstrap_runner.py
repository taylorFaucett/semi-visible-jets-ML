import os
from nn import nn
from sklearn.model_selection import ShuffleSplit
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
from average_decision_ordering import calc_ado
from tqdm import tqdm, trange

# Suppress all the tensorflow debugging info for new networks
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import pathlib

path = pathlib.Path.cwd()


def bootstrap_runner(run_number):
    number_path = bs_model_dir.parent.parent / f"p{run_number}"
    X = np.load(number_path / "X.npy")
    y = np.load(number_path / "y.npy")
    ll = np.load(path.parent / "data" / "raw" / "ll_predictions.npy")

    n = len(y)
    n_train = int(0.85 * n)
    n_test = int(0.15 * n)
    rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.15)
    rs.get_n_splits(X)

    ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.15)
    straps = []
    aucs = []
    ados = []
    bs_count = 0
    t = tqdm(list(rs.split(X)))
    counter = 0
    for train_index, test_index in t:
        X_train = X[train_index]
        y_train = y[train_index]
        X_val = X[test_index]
        y_val = y[test_index]
        ll_val = ll[test_index]
        model_file = f"{bs_model_dir}/bs-{bs_count}.h5"
        if not os.path.isfile(model_file):
            model = nn(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                epochs=3000,
                batch_size=batch_size,
                layers=layers,
                nodes=nodes,
                model_file=model_file,
                verbose=0,
            )
        else:
            model = tf.keras.models.load_model(model_file)

        val_predictions = np.hstack(model.predict(X_val))
        auc_val = roc_auc_score(y_val, val_predictions)
        ado_val = calc_ado(fx=val_predictions, gx=np.hstack(ll_val), target=y_val, n_pairs=1000000)
        straps.append(bs_count)
        aucs.append(auc_val)
        ados.append(ado_val)
        
        results = pd.DataFrame({"bs": straps, "auc": aucs, "ado":ados})
        results.to_csv(
            path
            / "runs"
            / "run-hl-jetPT-ircSafe-superSmallSet-smallKB-8Cones"
            / f"bootstrap_results_{run_number}.csv"
        )
        bs_count += 1
        auc_mean = np.average(aucs)
        auc_std = np.std(aucs)
        
        ado_mean = np.average(ados)
        ado_std = np.std(ados)
        
        counter += 1
        t.set_description(f"Run ({counter}/{n_splits}): (AUC = {auc_mean:.5f} +/- {auc_std:.5f} , ADO = {ado_mean:.5f} +/- {ado_std:.5f}")
        t.refresh() # to show immediately the update


if __name__ == "__main__":
    n_splits = 200  # number of bootstrap passes
    layers = 5
    nodes = 50
    batch_size = 256
    runs = [10]
    for run_number in runs:
        # Create a directory for boot-strap models if it doesn't exist
        bs_model_dir = (
            path
            / "runs"
            / "run-hl-jetPT-ircSafe-superSmallSet-smallKB-8Cones"
            / "bs_models"
            / f"p{run_number}"
        )
        if not bs_model_dir.exists():
            os.mkdir(bs_model_dir)

        # Run bootstrap
        bootstrap_runner(run_number)
