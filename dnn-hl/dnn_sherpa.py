from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
import os
import matplotlib.pyplot as plt
import sherpa
import sherpa.algorithms.bayesian_optimization as bayesian_optimization

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

import pathlib
path = pathlib.Path.cwd()

parameters = [sherpa.Continuous('learning_rate', [1e-4, 1e-2]),
              sherpa.Discrete('num_units', [32, 128]),
              sherpa.Choice('activation', ['relu', 'tanh', 'sigmoid'])]
algorithm = bayesian_optimization.GPyOpt(max_num_trials=50)
study = sherpa.Study(parameters=parameters,
                     algorithm=algorithm,
                     lower_is_better=False)

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def scale_data(x, mean=True):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        std = 1
    return (x-mean)/std

def get_data(rinv, N=None):        
    hl_file = path.parent / "data" / "jss_observables" / f"HL-{rinv}.h5"
    x = pd.read_hdf(hl_file, "features")
    y = pd.read_hdf(hl_file, "targets")

    if N is not None:
        x = x.loc[:N-1]
        y = y.loc[:N-1]
    for observable in list(x.columns):
        if "c2" in observable or "c3" in observable or "d2" in observable:
            x[observable] = np.log10(1.0+x[observable])

    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)
    return x, y


def run_sherpa(X, y, rinv):
    # To retrain, remove the old model
    model_file = path / "models" / f"{rinv}.h5"
    epochs = 15
    
    # Split data for train, test, validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )



    for trial in study:
        lr = trial.parameters['learning_rate']
        num_units = trial.parameters['num_units']
        act = trial.parameters['activation']

        # Create model
        model = Sequential([Flatten(input_shape=(X_train.shape[1], )),
                            Dense(num_units, activation=act),
                            Dense(1, activation='softmax')])
        
        
        optimizer = Adam(lr=lr)
        model.compile(loss='binary_crossentropy',
                      optimizer=optimizer,
                      metrics=['accuracy'])

        # Train model
        for i in range(epochs):
            model.fit(X_train, y_train)
            loss, accuracy = model.evaluate(x_test, y_test)
            study.add_observation(trial=trial, iteration=i,
                                  objective=accuracy,
                                  context={'loss': loss})
            if study.should_trial_stop(trial):
                break 
        
        study.finalize(trial=trial)    





    
if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    layers = 5
    nodes = 200
    for rinv in rinvs:
        # Grab jet images and labels
        X, y = get_data(rinv, N=20000)

        # Train a new model (or load the existing one if available)
        run_sherpa(X, y, rinv)
    