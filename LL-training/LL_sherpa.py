# Standard Imports
import pandas as pd
import h5py
import numpy as np
import sherpa
import tqdm
from sklearn.model_selection import train_test_split
import pathlib
path = pathlib.Path.cwd()

# Import homemade tools
from process_data import process_data
from cnn_model import cnn_model
from tts import tts


def setup_sherpa(max_num_trials):
    parameters = [
        sherpa.Continuous("learning_rate", [1e-4, 1e-2], "log"),
        sherpa.Continuous("dropout_1", [0, 0.5]),
        sherpa.Continuous("dropout_2", [0, 0.5]),
        sherpa.Ordinal(name='batch_size', range=[32, 64, 128]),
        sherpa.Discrete("filter_1", [20, 200]),
        sherpa.Discrete("filter_2", [20, 200]),
        sherpa.Discrete("filter_3", [20, 200]),
        sherpa.Discrete("kernel_1", [2,4]),
        sherpa.Discrete("kernel_2", [2,4]),
        sherpa.Discrete("kernel_3", [2,4]),
        sherpa.Discrete("dense_units_1", [32, 400]),
        sherpa.Discrete("dense_units_2", [32, 400]),
        sherpa.Discrete("dense_units_3", [32, 400]),
        sherpa.Discrete("pool_1", [2,3]), 
        sherpa.Discrete("pool_2", [2,3]), 
    ]

    algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(max_num_trials=max_num_trials)
    return parameters, algorithm

def get_data(rinv, N):
    df = h5py.File(path.parent / "data" / "jet_images" / f"LL-{rinv}.h5", "r")
    y = df["targets"][:N]
    X = df["features"][:N]
    X = process_data(X)
    return X, y

def run_sherpa(rinv):
    sherpa_results_file = path / "sherpa_results" / f"{rinv}.npy"
    if sherpa_results_file.exists():
        return sherpa_results_file

    parameters, algorithm = setup_sherpa(max_num_trials)
    study = sherpa.Study(parameters=parameters, algorithm=algorithm, lower_is_better=False)
    trial_epochs = 15
        
    X, y = get_data(rinv, N)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)
    t = tqdm.tqdm(study, total=max_num_trials)
    for trial in t:
        # Sherpa settings in trials
        tp = trial.parameters
        model = cnn_model(tp)
        for i in range(trial_epochs):
            model.fit(X_train, 
                      y_train, 
                      batch_size=int(tp["batch_size"]), 
                      verbose=0
                     )
            
            loss, accuracy, auc = model.evaluate(X_test, y_test, verbose = 0)
            study.add_observation(
                trial=trial,
                iteration=i,
                objective=auc,
                context={"loss": loss, "auc": auc, "accuracy":accuracy},
            )
            if study.should_trial_stop(trial):
                study.finalize(trial=trial, status="STOPPED")
                break

        t.set_description(f"Trial {trial.id}; rinv={rinv.replace('p','.')} -> Test AUC = {auc:.4}")
        study.finalize(trial=trial, status="COMPLETED")
        np.save(sherpa_results_file, study.get_best_result())

if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    N = 50000
    max_num_trials = 200
    
    for rinv in rinvs:
        run_sherpa(rinv)