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
from process_data import process_data
from cnn_model import cnn_model
from plot_roc import plot_roc

def test_val_set(rinv, split):
    f = h5py.File(path.parent / "data" / "jet_images" / f"LL-{rinv}.h5", "r")
    N = f["targets"].shape[0]
    if split == "val":
        a = int(N*0.8)
        b = int(N*0.9)
    if split == "test":
        a = int(N*0.9)
        b = -1
    X = f["features"][a:b]
    y = f["targets"][a:b]
    X = process_data(X)
    return N, X, y


def generator(rinv, batch_size, fullSet=False):
    while True:
        f = h5py.File(path.parent / "data" / "jet_images" / f"LL-{rinv}.h5", "r")
        N = f["targets"].shape[0]
        start = 0
        if not fullSet:
            N = int(N * 0.80)
        end = batch_size
        while start < N:
            X = f["features"][start:end]
            X = process_data(X)
            y = f["targets"][start:end]
            yield X, y
            start += batch_size
            end += batch_size


def train_cnn(rinv, retrain=False):
    # To retrain, remove the old model
    model_file = path / "models" / f"{rinv}.h5"
    if retrain and model_file.exists():
        os.remove(model_file)

    # Trainig parameters from the sherpa optimization
    tp = np.load(f"sherpa_results/{rinv}.npy", allow_pickle="TRUE").item()
    model = cnn_model(tp)
    
    training_data = generator(rinv=rinv, batch_size=tp["batch_size"])
    _, X_test, y_test = test_val_set(rinv=rinv, split="test")
    N_size, X_val, y_val = test_val_set(rinv=rinv, split="val")
    
    callbacks = [keras.callbacks.EarlyStopping(patience=10, verbose=1),
                 keras.callbacks.ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True),
                ]
    
    history = model.fit(
        training_data,
        epochs=200,
        verbose=2,
        steps_per_epoch = int(N_size * 0.80) // tp["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=callbacks
        )

    return model, X_test, y_test


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
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
