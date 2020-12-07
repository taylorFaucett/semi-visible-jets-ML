from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
import os
import matplotlib.pyplot as plt

import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.datasets import mnist
from keras.optimizers import Adam

import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization
import pathlib

path = pathlib.Path.cwd()

parameters = [
    sherpa.Continuous("learning_rate", [1e-4, 1e-2]),
    sherpa.Continuous("dropout", [0, 0.5]),
    sherpa.Discrete("num_units", [32, 128]),
    sherpa.Discrete("layers", [2, 8]),
    sherpa.Choice("activation", ["relu", "tanh", "sigmoid"]),
]
algorithm = bayesian_optimization.GPyOpt(max_num_trials=50)


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

    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)
    return x, y


def run_sherpa(X, y, rinv):
    # To retrain, remove the old model
    sherpa_results_file = path / "sherpa_results" / f"{rinv}.npy"
    if sherpa_results_file.exists():
        return True

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
        lr = trial.parameters["learning_rate"]
        num_units = trial.parameters["num_units"]
        activation = trial.parameters["activation"]
        dropout = trial.parameters["dropout"]
        layers = trial.parameters["layers"]
        optimizer = Adam(lr=lr)

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

        for i in range(epochs):
            model.fit(X_train, y_train)
            loss, accuracy = model.evaluate(X_test, y_test)
            study.add_observation(
                trial=trial, iteration=i, objective=accuracy, context={"loss": loss}
            )
            if study.should_trial_stop(trial):
                print("Stopping Trial {} after {} iterations.".format(trial.id, i + 1))
                study.finalize(trial=trial, status="STOPPED")
                break

        print("Saving optimized hyperparamter settings")
        study.finalize(trial=trial, status="COMPLETED")
        np.save(sherpa_results_file, study.get_best_result())


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        # Grab jet images and labels
        X, y = get_data(rinv, N=20000)

        # Train a new model (or load the existing one if available)
        run_sherpa(X, y, rinv)
