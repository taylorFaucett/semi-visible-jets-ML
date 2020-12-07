# Standard Imports
import pandas as pd
import h5py
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
import pathlib

path = pathlib.Path.cwd()

# Import homemade tools
from get_data import get_data
from HL_model import HL_model
from plot_roc import plot_roc


def train_HL(rinv, retrain=False):
    # To retrain, remove the old model
    model_file = path / "models" / f"{rinv}.h5"
    if retrain and model_file.exists():
        os.remove(model_file)

    # Trainig parameters from the sherpa optimization
    tp = np.load(f"sherpa_results/{rinv}.npy", allow_pickle="TRUE").item()

    X, y = get_data(rinv)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )

    model = HL_model(tp=tp, input_shape=X_train.shape[1])
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=model_file, verbose=1, save_best_only=True
        ),
    ]

    history = model.fit(
        X_train,
        y_train,
        epochs=200,
        verbose=2,
        batch_size=int(tp["batch_size"]),
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    return model, X_test, y_test


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        # Train a new model (or load the existing one if available)
        model, X_test, y_test = train_HL(rinv, retrain=False)

        # Plot the ROC curve
        test_predictions = model.predict(X_test).ravel()
        auc_val = plot_roc(y_test, test_predictions, rinv)
        print(rinv, auc_val)
