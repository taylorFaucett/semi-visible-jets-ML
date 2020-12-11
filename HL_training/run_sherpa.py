# Standard Imports
import pandas as pd
import h5py
import numpy as np
import sherpa
import tqdm
from sklearn.model_selection import train_test_split
import pathlib

# Import homemade tools
from get_data import get_data
from get_model import get_model

path = pathlib.Path.cwd()


def setup_sherpa(max_num_trials):
    parameters = [
        sherpa.Continuous("learning_rate", [1e-4, 1e-2], "log"),
        sherpa.Continuous("dropout_1", [0, 0.5]),
        sherpa.Continuous("dropout_2", [0, 0.5]),
        sherpa.Ordinal(name="batch_size", range=[256, 512, 1024]),
        sherpa.Discrete("dense_units_1", [32, 400]),
        sherpa.Discrete("dense_units_2", [32, 400]),
        sherpa.Discrete("dense_units_3", [32, 400]),
    ]

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
        max_num_trials=max_num_trials
    )
    return parameters, algorithm


def run_sherpa(rinv):
    sherpa_results_file = path / "sherpa_results" / f"{rinv}.npy"
    if sherpa_results_file.exists():
        return sherpa_results_file

    parameters, algorithm = setup_sherpa(max_num_trials)
    study = sherpa.Study(
        parameters=parameters, algorithm=algorithm, lower_is_better=False
    )

    X, y = get_data(rinv, N)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.8, random_state=42
    )
    t = tqdm.tqdm(study, total=max_num_trials)
    for trial in t:
        # Sherpa settings in trials
        tp = trial.parameters
        model = get_model(tp=tp, input_shape=X_train.shape[1])
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

        t.set_description(
            f"Trial {trial.id}; rinv={rinv.replace('p','.')} -> Test AUC = {auc:.4}"
        )
        np.save(sherpa_results_file, study.get_best_result())
        study.finalize(trial=trial, status="COMPLETED")


if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    N = 200000
    max_num_trials = 100
    trial_epochs = 15

    for rinv in rinvs:
        run_sherpa(rinv)
