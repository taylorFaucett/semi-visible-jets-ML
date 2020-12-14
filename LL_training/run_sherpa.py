# Standard Imports
import os
import pandas as pd
import h5py
import numpy as np
import sherpa
import tqdm
from sklearn.model_selection import train_test_split
import pathlib

# Import homemade tools
from process_data import process_data
from get_model import get_model
from get_data import get_data

path = pathlib.Path.cwd()

def run_sherpa(rinv):
    results_path = path / "sherpa_results" / rinv
    if not results_path.exists():
        os.mkdir(results_path)
        
    results_csv = results_path / "results.csv"
    if results_csv.exists():
        return
        
    parameters = [
        sherpa.Continuous("learning_rate", [1e-4, 1e-2], "log"),
        sherpa.Continuous("dropout_0", [0, 0.5]),
        sherpa.Continuous("dropout_1", [0, 0.5]),
        sherpa.Continuous("dropout_2", [0, 0.5]),
        sherpa.Ordinal("batch_size", [128, 256, 512]),
        sherpa.Discrete("filter_1", [16, 400]),
        sherpa.Discrete("filter_2", [16, 400]),
        sherpa.Discrete("filter_3", [16, 400]),
        sherpa.Discrete("dense_units_1", [20, 400]),
        sherpa.Discrete("dense_units_2", [20, 400]),
        sherpa.Discrete("dense_units_3", [20, 400]),
    ]

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(max_num_trials=max_num_trials)

    study = sherpa.Study(
        parameters=parameters, algorithm=algorithm, lower_is_better=False, disable_dashboard=True, output_dir=results_path
    )

    X, y = get_data(rinv, N)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.75, random_state=42
    )
    t = tqdm.tqdm(study, total=max_num_trials)
    for tix, trial in enumerate(t):
        # Sherpa settings in trials
        tp = trial.parameters
        model = get_model(tp)
        for i in range(trial_epochs):
            training_error = model.fit(X_train, y_train, batch_size=int(tp["batch_size"]), verbose=0)
            loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
            study.add_observation(
                trial=trial,
                iteration=i,
                objective=auc,
                context={"loss": loss, 
                         "auc":auc,
                         "accuracy": accuracy},
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
    rinvs = ["0p0", "0p3", "1p0"]
    N = 100000
    max_num_trials = 100
    trial_epochs = 10
    for rinv in rinvs:
        run_sherpa(rinv)
