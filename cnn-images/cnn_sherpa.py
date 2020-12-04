# Standard Imports
import pandas as pd
import h5py
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# Keras Imports
import keras
from keras.models import Sequential
from keras import metrics
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import plot_model
import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization

# Path Imports
import pathlib

path = pathlib.Path.cwd()

def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x - mean) / std


def get_data(rinv, N):
    df = h5py.File(path.parent / "data" / "jet_images" / f"LL-{rinv}.h5", "r")
    y = df["targets"][:N]
    x = df["features"][:N]
    # x = np.log10(1.0 + x)
    x = scale_data(x, mean=True)
    x = np.expand_dims(x, axis=3)
    return x, y

def run_sherpa(X, y, rinv):
    sherpa_results_file = path / "sherpa_results" / f"{rinv}.npy"
    if sherpa_results_file.exists():
        return sherpa_results_file
    
    parameters = [
        sherpa.Continuous("learning_rate", [1e-4, 1e-2]),
        sherpa.Continuous("dropout", [0, 0.5]),
        sherpa.Discrete("filters", [16, 128]),
        sherpa.Discrete("dense_units", [32, 200]),
        sherpa.Discrete("conv_blocks", [2, 3]),
        sherpa.Discrete("dense_layers", [2, 5]),
        sherpa.Ordinal(name='batch_size', range=[16, 32, 64, 128]),
        sherpa.Ordinal("pool_1", [2]), #[2,4]
        sherpa.Ordinal("pool_2", [2]), #[2,4]
        sherpa.Ordinal("kernel_1", [3]), #[2,3]
        sherpa.Ordinal("kernel_2", [3]), #[2,3]
        sherpa.Choice("activation", ["relu"]),
    ]
    
    algorithm = bayesian_optimization.GPyOpt(max_num_trials=50)

    study = sherpa.Study(
        parameters=parameters, algorithm=algorithm, lower_is_better=False
    )
    epochs = 25

    # Split data for train, test, validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )

    for trial in study:
        # Sherpa settings in trials
        learning_rate = trial.parameters["learning_rate"]
        batch_size = trial.parameters["batch_size"]
        filters = trial.parameters["filters"]
        dense_units = trial.parameters["dense_units"]
        activation = trial.parameters["activation"]
        dropout = trial.parameters["dropout"]
        conv_blocks = trial.parameters["conv_blocks"]
        dense_layers = trial.parameters["dense_layers"]
        pool_1 = trial.parameters["pool_1"]
        pool_2 = trial.parameters["pool_2"]
        kernel_1 = trial.parameters["kernel_1"]
        kernel_2 = trial.parameters["kernel_2"]
        optimizer = Adam(lr=learning_rate)

        model = Sequential()
        for lix in range(conv_blocks):
            model.add(
                Conv2D(
                    filters,
                    (kernel_1, kernel_2),
                    activation=activation,
                    padding="same",
                    input_shape=(32, 32, 1),
                )
            )
            model.add(MaxPooling2D((pool_1, pool_2), padding="same"))
        model.add(Flatten())
        for lix in range(dense_layers):
            model.add(Dense(dense_units, activation=activation))
            if lix <= dense_layers - 2:
                model.add(Dropout(dropout))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy", metrics.AUC(name="auc")],
        )
        for i in range(epochs):
            model_plot_file = (
                path
                / "sherpa_results"
                / "model_images"
                / rinv
                / f"model_{trial.id}.png"
            )
            model.fit(X_train, y_train, batch_size=batch_size)
            loss, accuracy, auc = model.evaluate(X_test, y_test)
            study.add_observation(
                trial=trial,
                iteration=i,
                objective=accuracy,
                context={"loss": loss, "AUC": auc},
            )
            # if not model_plot_file.exists():
            plot_model(
                model, to_file=model_plot_file, show_shapes=True,
            )

            if study.should_trial_stop(trial):
                print("Stopping Trial {} after {} iterations.".format(trial.id, i + 1))
                study.finalize(trial=trial, status="STOPPED")
                break

        print("Saving optimized hyperparamter settings")
        study.finalize(trial=trial, status="COMPLETED")
        np.save(sherpa_results_file, study.get_best_result())
    return sherpa_results_file


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    N = 25000
    for rinv in rinvs:
        # Grab jet images and labels
        X, y = get_data(rinv, N)

        # Train a new model (or load the existing one if available)
        results_file = run_sherpa(X, y, rinv)
        sherpa_selection = np.load(results_file, allow_pickle="TRUE").item()
        print(rinv, sherpa_selection)
