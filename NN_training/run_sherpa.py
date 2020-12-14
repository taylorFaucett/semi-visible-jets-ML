# Standard Imports
import os
import sys
import pandas as pd
import h5py
import numpy as np
import sherpa
import tqdm
from sklearn.model_selection import train_test_split
import pathlib

# Import homemade tools
from get_model import get_model
from get_data import get_data

path = pathlib.Path.cwd()


def get_param(run_type):
    if run_type == "LL":
        parameters = [
            sherpa.Continuous("learning_rate", [1e-4, 1e-2], "log"),
            # sherpa.Continuous("dropout_0", [0, 0.5]),
            # sherpa.Continuous("dropout_1", [0, 0.5]),
            # sherpa.Continuous("dropout_2", [0, 0.5]),
            sherpa.Ordinal("batch_size", [128, 256, 512]),
            sherpa.Discrete("filter_1", [16, 400]),
            sherpa.Discrete("filter_2", [16, 400]),
            sherpa.Discrete("filter_3", [16, 400]),
            sherpa.Discrete("dense_units_1", [20, 400]),
            sherpa.Discrete("dense_units_2", [20, 400]),
            sherpa.Discrete("dense_units_3", [20, 400]),
        ]
    elif run_type == "HL":
        parameters = [
            sherpa.Continuous("learning_rate", [1e-4, 1e-2], "log"),
            # sherpa.Continuous("dropout_0", [0, 0.5]),
            # sherpa.Continuous("dropout_1", [0, 0.5]),
            # sherpa.Continuous("dropout_2", [0, 0.5]),
            sherpa.Ordinal("batch_size", [128, 256, 512]),
            sherpa.Discrete("dense_units_1", [20, 400]),
            sherpa.Discrete("dense_units_2", [20, 400]),
            sherpa.Discrete("dense_units_3", [20, 400]),
        ]
    return parameters


def run_sherpa(run_type, rinv):
    results_path = path / "sherpa_results" / run_type / rinv
    if not results_path.parent.exists():
        os.mkdir(results_path.parent)
    if not results_path.exists():
        os.mkdir(results_path)

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
        max_num_trials=max_num_trials
    )

    parameters = get_param(run_type)

    study = sherpa.Study(
        parameters=parameters,
        algorithm=algorithm,
        lower_is_better=False,
        disable_dashboard=True,
        output_dir=results_path,
    )

    X, y = get_data(run_type, rinv, N)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.75, random_state=42
    )
    t = tqdm.tqdm(study, total=max_num_trials)
    for trial in t:
        # Sherpa settings in trials
        tp = trial.parameters
        if run_type == "HL":
            input_shape = X_train.shape[1]
        else:
            input_shape = None
        model = get_model(run_type, tp, input_shape=input_shape)
        for i in range(trial_epochs):
            model.fit(X_train, y_train, batch_size=int(tp["batch_size"]), verbose=0)
            loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
            study.add_observation(
                trial=trial,
                iteration=i,
                objective=auc,
                context={"loss": loss, "auc": auc, "accuracy": accuracy},
            )
            if study.should_trial_stop(trial):
                study.finalize(trial=trial, status="STOPPED")
                break

        study.finalize(trial=trial, status="COMPLETED")
        np.save(results_path / "best_results.npy", study.get_best_result())
        study.save()
        t.set_description(
            f"Trial {trial.id}; rinv={rinv.replace('p','.')} -> AUC = {auc:.4}"
        )


if __name__ == "__main__":
    run_type = str(sys.argv[1])
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        N = 100000
        max_num_trials = 50
        trial_epochs = 15
        run_sherpa(run_type, rinv)
