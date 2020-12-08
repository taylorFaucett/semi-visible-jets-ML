from guided_iteration import data_grabber
from sklearn.model_selection import KFold
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np
import shutil
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

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

    lr_init = 0.5

    def scheduler(epoch):
        if epoch < 30:
            return lr_init
        else:
            return lr_init * tf.math.exp(0.05 * (30 - epoch))

    # print("    Training a new model at: " + model_file)
    model = tf.keras.Sequential()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=True,
        name="Adam",
    )
    
    initializer = tf.keras.initializers.GlorotUniform()

#     optimizer = tf.keras.optimizers.SGD(
#         learning_rate=lr_init, momentum=0.0, nesterov=False, name="SGD"
#     )

    model.add(tf.keras.layers.Flatten(input_dim=X_train.shape[1]))
    for lix in range(layers):
        model.add(
            tf.keras.layers.Dense(
                nodes,
                kernel_initializer=initializer,
                activation="relu",
                kernel_constraint=tf.keras.constraints.MaxNorm(3),
                bias_constraint=tf.keras.constraints.MaxNorm(3),
#                 kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-4),
#                 bias_regularizer=tf.keras.regularizers.l2(1e-4),
#                 activity_regularizer=tf.keras.regularizers.l2(1e-5)
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

    lrs = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=verbose)

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
    home = "/home/tfaucett/Projects/ado-iteration"
    data_dir = f"{home}/data"
    efp_dir = f"{data_dir}/efp"
    it_dir = f"{home}/guided-iteration/results/{run_name}"
    incl_hl = True

    model_dir = f"{it_dir}/k_models"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    k_fold_file = f"{home}/guided-iteration/k_fold_nn.py"
    k_fold_copy = f"{it_dir}/k_fold_copy.py"
    shutil.copy(k_fold_file, k_fold_copy)
    
    # Get selected efps
    efps = pd.read_csv(f"{it_dir}/selected_efps.csv").efp.values[1:]
    print(efps)
    X, y = data_grabber(
        selected_efps=efps,
        data_dir=data_dir,
        efp_dir=efp_dir,
        incl_hl=False, 
        incl_pt=True, 
        incl_mass_only=True,
        normalize=False,
    )   
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
        if not os.path.isfile(model_file):
            model = nn(
                X_train=X_train,
                y_train=y_train,
                X_val=X_test,
                y_val=y_test,
                epochs=1000,
                batch_size=1024, #32
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
    run_name = "black-box-noNN"
    n_folds = 10
    k_fold_nn(run_name, n_folds)
