from brute_force_iteration import data_grabber
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import shutil
from os import path, getcwd, environ, mkdir

environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


def nn(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    layers,
    nodes,
    ix,
    model_file,
    verbose,
):

    # print("    Training a new model at: " + model_file)
    model = tf.keras.Sequential()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    initializer = tf.keras.initializers.GlorotUniform()

    model.add(tf.keras.layers.Flatten(input_dim=X_train.shape[1]))
    for lix in range(layers):
        model.add(
            tf.keras.layers.Dense(
                nodes,
                kernel_initializer=initializer,
                activation="relu",
                kernel_constraint=tf.keras.constraints.MaxNorm(3),
                bias_constraint=tf.keras.constraints.MaxNorm(3),
            )
        )
        if lix <= layers - 2:
            model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",  # binary_crossentropy
        optimizer=optimizer,
        metrics=[
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Accuracy(name="acc"),
        ],
    )
    mc = tf.keras.callbacks.ModelCheckpoint(
        model_file, monitor="val_auc", verbose=verbose, save_best_only=True, mode="max",
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", verbose=verbose, patience=25
    )

    callbacks = [mc, es]

    if verbose > 0:
        print(model.summary())

    model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    return model


def k_fold_nn(run_name, n_folds):
    home = path.dirname(getcwd())
    data_dir = path.join(home, "data")
    efp_dir = path.join(data_dir, "efp")
    it_dir = path.join(home, "brute-force")
    model_dir = path.join(it_dir, "k_models")
    if not path.exists(model_dir):
        mkdir(model_dir)

    # Get selected efps
    selected_efps = path.join(it_dir, "selected_efps.csv")
    efps = pd.read_csv(selected_efps).efp.values
    print(efps)
    X, y = data_grabber(efps)
    X = X.to_numpy()
    kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)
    kf.get_n_splits(X)
    # KFold(n_splits=n_folds, random_state=None, shuffle=False)
    kix = 0
    aucs = np.zeros(n_folds)
    for train_index, test_index in kf.split(X):
        # print("TRAIN:", train_index, "TEST:", test_index)
        model_file = f"{model_dir}/k{kix}.h5"
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if not path.isfile(model_file):
            model = nn(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=1000,
                batch_size=512,  # 32
                layers=3,
                nodes=300,
                ix=kix,
                model_file=model_file,
                verbose=0,
            )
        else:
            model = tf.keras.models.load_model(model_file)
        yhat = np.hstack(model.predict(X_test))
        auc_val = roc_auc_score(y_test, yhat)
        aucs[kix] = auc_val
        print(f"pass {kix} -> test-set AUC={auc_val:.4}")
        results = pd.DataFrame({"auc": aucs})
        results.to_csv(f"{it_dir}/k_fold_results.csv")
        kix += 1
    print(aucs)
    auc_avg = np.average(aucs)
    auc_std = np.std(aucs)
    print(f"AUC = {auc_avg} ± {auc_std}")


if __name__ == "__main__":
    run_name = "black-box-KS"
    n_folds = 10
    k_fold_nn(run_name, n_folds)
