# Standard Imports
import os
import sys
import pandas as pd
import h5py
import numpy as np
import sherpa
from tqdm import tqdm
from sklearn.model_selection import train_test_split, ShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve
import pathlib
import pickle

# Keras/TF imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import metrics
from keras.initializers import Orthogonal
from keras.constraints import max_norm
from keras.regularizers import l2

# Import homemade tools
import getData

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


kernel_initializer = Orthogonal(gain=1.0, seed=None)
kernel_constraint = max_norm(3)
bias_constraint = max_norm(3)

path = pathlib.Path.cwd()


class NN_training:
    def __init__(self, run_type, rinv):
        self.run_type = run_type
        self.rinv = rinv
        self.N = 500
        self.max_num_trials = 2
        self.trial_epochs = 10
        self.sherpaTrained = False
        self.bootstrapTrained = False

    @classmethod
    def load(cls, pkl_file):
        return pickle.load(open(pkl_file, "rb"))

    def save(self, pkl_file):
        dbfile = open(pkl_file, "wb")
        pickle.dump(self, dbfile)
        dbfile.close()

    def getParam(self):
        if self.run_type == "LL":
            parameters = [
                sherpa.Continuous("learning_rate", [1e-5, 1e-3], "log"),
                sherpa.Continuous("dropout", [0.1, 0.5]),
                sherpa.Discrete("conv_blocks", [1, 2]),
                sherpa.Discrete("filter_units", [16, 128]),
                sherpa.Discrete("dense_units", [25, 200]),
                sherpa.Discrete("dense_layers", [2, 5]),
            ]
        elif self.run_type == "HL":
            parameters = [
                sherpa.Continuous("learning_rate", [1e-5, 1e-3], "log"),
                sherpa.Continuous("dropout", [0, 0.5]),
                sherpa.Ordinal("dense_layers", [2, 8]),
                sherpa.Discrete("dense_units", [1, 200]),
            ]
        return parameters

    def getModel(self):
        if self.run_type == "LL":
            model = self.LL_model()
        elif self.run_type == "HL":
            model = self.HL_model()
        return model

    def LL_model(self):
        model = Sequential()
        model.add(
            Conv2D(
                32,
                kernel_size=3,
                padding="same",
                activation="relu",
                input_shape=self.input_shape,
                kernel_initializer=kernel_initializer,
                kernel_constraint=kernel_constraint,
                bias_constraint=kernel_constraint,
            )
        )
        for filter_ix in range(int(self.tp["conv_blocks"])):
            for conv_ix in range(2):
                model.add(
                    Conv2D(
                        int(self.tp["filter_units"]),
                        kernel_size=3,
                        padding="same",
                        activation="relu",
                        kernel_initializer=kernel_initializer,
                        kernel_constraint=kernel_constraint,
                        bias_constraint=kernel_constraint,
                    )
                )
            model.add(MaxPooling2D(pool_size=2))
        model.add(Flatten())
        for dense_ix in range(int(self.tp["dense_layers"])):
            model.add(
                Dense(
                    self.tp["dense_units"],
                    activation="relu",
                    kernel_initializer=kernel_initializer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=kernel_constraint,
                )
            )
            if dense_ix - 1 < int(self.tp["dense_layers"]):
                model.add(Dropout(self.tp["dropout"]))

        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(lr=self.tp["learning_rate"]),
            metrics=["accuracy", metrics.AUC(name="auc")],
        )
        return model

    def HL_model(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.input_shape,))
        for dense_ix in range(int(self.tp["dense_layers"])):
            model.add(
                Dense(
                    self.tp["dense_units"],
                    activation="relu",
                    kernel_initializer=kernel_initializer,
                    kernel_constraint=kernel_constraint,
                    bias_constraint=kernel_constraint,
                )
            )
            if dense_ix - 1 < int(self.tp["dense_layers"]):
                model.add(Dropout(self.tp["dropout"]))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            loss="binary_crossentropy",
            optimizer=Adam(lr=self.tp["learning_rate"]),
            metrics=["accuracy", metrics.AUC(name="auc")],
        )

    def bestSherpa(self, objective="Objective"):
        # Use sherpa parameters (same as used in bootstrapping)
        sherpa_file = (
            path / "results" / "sherpa" / self.run_type / self.rinv / "results.csv"
        )
        sherpa_results = pd.read_csv(sherpa_file, index_col="Trial-ID").groupby(
            "Status"
        )
        sorted_results = sherpa_results.get_group("COMPLETED").sort_values(
            by=objective, ascending=False
        )
        best_result = sorted_results.iloc[0].to_dict()
        return best_result

    def sherpa(self):
        algorithm = sherpa.algorithms.bayesian_optimization.GPyOpt(
            max_num_trials=self.max_num_trials
        )
        parameters = self.getParam()

        results_path = path / "results" / "sherpa" / self.run_type / self.rinv
        results_path.mkdir(parents=True, exist_ok=True)

        study = sherpa.Study(
            parameters=parameters,
            algorithm=algorithm,
            lower_is_better=False,
            disable_dashboard=True,
            output_dir=results_path,
        )

        data = getData.getData(self.run_type, self.rinv, self.N)
        X = data.features()
        y = data.targets()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.75, random_state=42
        )
        t = tqdm(study, total=self.max_num_trials)
        for trial in t:
            # Sherpa settings in trials
            self.tp = trial.parameters
            self.input_shape = X_train.shape[1:]
            model = self.getModel()

            for i in range(self.trial_epochs):
                model.fit(X_train, y_train, epochs=3, batch_size=128, verbose=0)
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
            study.save()
            t.set_description(
                f"Trial {trial.id}; rinv={self.rinv.replace('p','.')} -> AUC = {auc:.4}"
            )
            self.sherpaTrained = True

    def bootstrap(self, n_splits=200):
        # Trainig parameters from the sherpa optimization
        self.tp = self.bestSherpa()
        rs = ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.10)

        data = getData.getData(self.run_type, self.rinv, self.N)
        X = data.features()
        y = data.targets()

        rs.get_n_splits(X)
        ShuffleSplit(n_splits=n_splits, random_state=0, test_size=0.10)
        straps = []
        aucs = []
        boot_ix = 0
        t = tqdm(list(rs.split(X)))

        for train_index, test_index in t:
            X_train = X[train_index]
            y_train = y[train_index]
            X_val = X[test_index]
            y_val = y[test_index]

            bs_path = path / "results" / "bootstrap" / self.run_type / self.rinv
            model_file = bs_path / "models" / f"bs_{boot_ix}.h5"
            roc_file = bs_path / "roc" / f"roc_{boot_ix}.csv"
            ll_pred_file = bs_path / "ll_predictions" / f"pred_{boot_ix}.npy"

            model_file.parent.mkdir(parents=True, exist_ok=True)
            roc_file.parent.mkdir(parents=True, exist_ok=True)
            ll_pred_file.parent.mkdir(parents=True, exist_ok=True)

            if not model_file.exists():
                model = self.getModel()
                callbacks = [
                    keras.callbacks.EarlyStopping(
                        monitor="val_auc",
                        patience=3,
                        min_delta=0.0001,
                        verbose=0,
                        restore_best_weights=True,
                        mode="max",
                    ),
                    keras.callbacks.ModelCheckpoint(
                        filepath=model_file, verbose=0, save_best_only=True
                    ),
                ]

                model.fit(
                    X_train,
                    y_train,
                    epochs=200,
                    verbose=0,
                    batch_size=128,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                )

            else:
                model = keras.models.load_model(model_file)

            val_predictions = np.hstack(model.predict(X_val))
            auc_val = roc_auc_score(y_val, val_predictions)

            # Save the predictions

            np.save(ll_pred_file, model.predict(X))
            straps.append(boot_ix)
            aucs.append(auc_val)

            fpr, tpr, _ = roc_curve(y_val, val_predictions)
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

            roc_df.to_csv(roc_file)

            results = pd.DataFrame({"bs": straps, "auc": aucs})
            results.to_csv(bs_path / "aucs.csv")
            auc_mean, auc_ci = self.mean_ci(aucs)
            boot_ix += 1
            t.set_description(
                f"rinv={rinv} ({boot_ix}/{n_splits}): (AUC = {auc_mean:.4f} +/- {auc_ci:.4f})"
            )
            t.refresh()
        self.bootstrapTrained = True

    def mean_ci(data, confidence=0.95):
        a = 1.0 * np.array(data)
        n = len(a)
        m, se = np.mean(a), scipy.stats.sem(a)
        h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return m, h


if __name__ == "__main__":
    run_types = ["LL"]
    rinvs = ["0p0", "0p3", "1p0"]
    for run_type in run_types:
        for rinv in rinvs:
            run_file = path / "results" / "pkl" / run_type / f"{rinv}.pkl"
            run_file.parent.mkdir(parents=True, exist_ok=True)
            run = NN_training(run_type, rinv)
            if run_file.exists():
                print("Loading...")
                run = run.load(run_file)
            print(run.sherpaTrained)
            if not run.sherpaTrained:
                run.sherpa()
                run.save(run_file)
            if not hasattr(run_file, "bootstrapTrained"):
                run.bootstrap(n_splits=5)
                run.save(run_file)
