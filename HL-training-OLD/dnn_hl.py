import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
import os
import matplotlib.pyplot as plt

import pathlib

path = pathlib.Path.cwd()


def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x - mean) / std


def get_data(rinv, N=None):
    hl_file = path.parent / "data" / "jss_observables" / f"HL-{rinv}.h5"
    x = pd.read_hdf(hl_file, "features")
    y = pd.read_hdf(hl_file, "targets")
    if N is not None:
        x = x.loc[: N - 1]
        y = y.loc[: N - 1]
    # for observable in list(x.columns):
    #     if "c2" in observable or "c3" in observable or "d2" in observable:
    #         x[observable] = np.log10(1.0 + x[observable])

    #     x = scale_data(x.to_numpy(), mean=True)
    observable_list = list(x.columns)
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)
    return x, y, observable_list


def plot_roc(X_test, y_test, rinv):
    auc_save_file = path / "roc_df" / f"auc-{rinv}.txt"
    test_predictions = model.predict(X_test).ravel()
    auc = metrics.roc_auc_score(y_test, test_predictions)
    with open(auc_save_file, "w") as f:
        f.write(str(auc))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predictions)
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
    roc_df.to_csv(f"roc_df/{rinv}.csv")
    # background_rejection = 1./fpr
    rinv_str = rinv.replace("p", ".")
    plt.plot(
        signal_efficiency,
        background_rejection,
        lw=2,
        label="$r_{inv} = %s$ (AUC $= %0.4f$)" % (rinv_str, auc),
    )
    #     plt.yscale("log")
    plt.xlabel("Signal efficiency $(\epsilon_S)$")
    plt.ylabel("Background rejection $(1 - \epsilon_B)$")
    #     plt.xlim([0,1])
    #     plt.ylim([0,1])
    plt.title("HL: " + ", ".join(observable_list))
    plt.legend(loc="lower left")
    plt.savefig(path / "figures" / "cnn_roc.png")
    plt.savefig(path / "figures" / "cnn_roc.pdf")
    return auc


def train_dnn(X, y, rinv, retrain=False):
    # To retrain, remove the old model
    model_file = path / "models" / f"{rinv}.h5"
    if retrain and model_file.exists():
        os.remove(model_file)

    # Split data for train, test, validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )

    # Model parameters
    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1],)))

    for lix in range(layers):
        model.add(Dense(num_units, activation=activation))
        if lix <= layers - 2:
            model.add(Dropout(dropout))

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"],
    )

    callbacks = [
        EarlyStopping(patience=5),
        ModelCheckpoint(filepath=model_file),
        # TensorBoard(log_dir="./logs"),
    ]

    history = model.fit(
        X_train,
        y_train,
        batch_size=64,
        epochs=250,
        verbose=2,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    return model, X_test, y_test


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]

    for rinv in rinvs:
        # Import sherpa optimized hyperparamters for the rinv choice
        sherpa_results = np.load(
            f"sherpa_results/{rinv}.npy", allow_pickle="TRUE"
        ).item()
        print(sherpa_results)

        # Set network parameters
        learning_rate = sherpa_results["learning_rate"]
        activation = sherpa_results["activation"]
        num_units = sherpa_results["num_units"]
        dropout = sherpa_results["dropout"]
        layers = sherpa_results["layers"]
        optimizer = Adam(lr=learning_rate)
        

        # Grab jet images and labels
        X, y, observable_list = get_data(rinv=rinv)

        # Train a new model (or load the existing one if available)
        model, X_test, y_test = train_dnn(X, y, rinv, retrain=False)

        # Plot the ROC curve
        auc_val = plot_roc(X_test, y_test, rinv)

        # Generate predictions for the full dataset
        full_predictions = np.concatenate(model.predict(X))

        # Save the predictions
        np.save(path / "predictions" / "ll_predictions.npy", full_predictions)
