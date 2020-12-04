# Standard Imports
import pandas as pd
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve

# Keras Imports
import keras
from keras import metrics
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Path Imports
import pathlib

path = pathlib.Path.cwd()


def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x - mean) / std


def test_val_set(rinv, split):
    f = h5py.File(path.parent / "data" / "jet_images" / f"LL-{rinv}.h5", "r")
    dSize = f["targets"].shape[0]
    train_size, val_size, test_size = (
        int(np.floor(dSize * 0.80)),
        int(np.floor(dSize * 0.1)),
        int(np.floor(dSize * 0.1)),
    )
    if split == "val":
        start = train_size
        end = train_size + val_size
    if split == "test":
        start = train_size + val_size
        end = -1

    X = f["features"][start:end]
    X = scale_data(X, mean=True)
    X = np.expand_dims(X, axis=3)
    y = f["targets"][start:end]
    return dSize, X, y


def generator(rinv, batch_size, fullSet=False):
    while True:
        f = h5py.File(path.parent / "data" / "jet_images" / f"LL-{rinv}.h5", "r")
        N = f["targets"].shape[0]
        start = 0
        if not fullSet:
            N = int(np.floor(N * 0.80))
            
        end = batch_size

        while start < N:
            X = f["features"][start:end]
            X = scale_data(X, mean=True)
            X = np.expand_dims(X, axis=3)
            y = f["targets"][start:end]
            yield X, y
            start += batch_size
            end += batch_size


def plot_roc(X_test, y_test, rinv):
    auc_save_file = path / "roc_df" / f"auc-{rinv}.txt"
    test_predictions = model.predict(X_test).ravel()
    auc = roc_auc_score(y_test, test_predictions)
    with open(auc_save_file, "w") as f:
        f.write(str(auc))

    fpr, tpr, thresholds = roc_curve(y_test, test_predictions)
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
        label="$r_{inv} = %s$ ($AUC = %0.3f$)" % (rinv_str, auc),
    )
    plt.xlabel("Signal efficiency $(\epsilon_S)$")
    plt.ylabel("Background rejection $(1 - \epsilon_B)$")
    plt.title("ROC - CNN on Jet Images")
    plt.legend(loc="lower left")
    plt.savefig(path / "figures" / "cnn_roc.png")
    plt.savefig(path / "figures" / "cnn_roc.pdf")
    return auc


def train_cnn(rinv, retrain=False):
    # To retrain, remove the old model
    model_file = path / "models" / f"{rinv}.h5"
    if retrain and model_file.exists():
        os.remove(model_file)

    # If the model already exists (i.e. we haven't removed the last one) train a new model

    optimizer = Adam(lr=0.001)

    model = Sequential()
    for lix in range(conv_layers):
        model.add(
            Conv2D(
                conv_units,
                (kernel_1, kernel_2),
                activation=activation,
                padding="same",
                input_shape=(32, 32, 1),
            )
        )
        model.add(MaxPooling2D((pool_1, pool_2), padding="same"))
    model.add(Flatten())
    for lix in range(dense_layers):
        model.add(Dense(dense_units, activation="relu"))
        if lix <= dLayers - 2:
            model.add(Dropout(dropout))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy", metrics.AUC(name="auc")],
    )
    callbacks = [
        EarlyStopping(patience=10, verbose=1),
        ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True),
    ]

    training_data = generator(rinv=rinv, batch_size=batch_size)
    N_size, X_test, y_test = test_val_set(rinv=rinv, split="test")
    N_size, X_val, y_val = test_val_set(rinv=rinv, split="val")
    
    history = model.fit(
        training_data,
        epochs=200,
        verbose=1,
        steps_per_epoch = (N_size * 0.80) // batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    return model, X_test, y_test


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]

    batch_size = 500
    for rinv in rinvs:
        # Import sherpa optimized hyperparamters for the rinv choice
        sherpa_results = np.load(
            f"sherpa_results/{rinv}.npy", allow_pickle="TRUE"
        ).item()

        # Set network parameters
        learning_rate = sherpa_results["learning_rate"]
        conv_units = sherpa_results["conv_units"]
        dense_units = sherpa_results["dense_units"]
        activation = sherpa_results["activation"]
        dropout = sherpa_results["dropout"]
        conv_layers = sherpa_results["conv_layers"]
        dense_layers = sherpa_results["dense_layers"]
        pool_1 = sherpa_results["pool_1"]
        pool_2 = sherpa_results["pool_2"]
        kernel_1 = sherpa_results["kernel_1"]
        kernel_2 = sherpa_results["kernel_2"]
        optimizer = Adam(lr=learning_rate)

        # Train a new model (or load the existing one if available)
        model, X_test, y_test = train_cnn(rinv, retrain=False)

        # Plot the ROC curve
        auc_val = plot_roc(X_test, y_test, rinv)
        print(rinv, auc_val)

        # Generate predictions for the full dataset
#         X = generator(rinv=rinv, batch_size=batch_size, fullSet=True)
#         full_predictions = model.predict(X) #np.concatenate(model.predict(X))
#         print(full_predictions)
        
#         # Save the predictions
#         np.save(path / "predictions" / "ll_predictions.npy", full_predictions)
