# standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports
import numpy as np
import h5py
import tensorflow as tf
from tqdm import tqdm

# energyflow imports
import energyflow as ef
from energyflow.archs import EFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, to_categorical

from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

import pathlib

path = pathlib.Path.cwd()


def run_EFN(X, Y, layer, lr):
    # network architecture parameters
    Phi_sizes, F_sizes = (100, 100, layer), (100, 100, 100)

    N = len(y)
    train, val, test = int(N * 0.75), int(N * 0.10), int(N * 0.15)

    # do train/val/test split
    (
        z_train,
        z_val,
        z_test,
        p_train,
        p_val,
        p_test,
        Y_train,
        Y_val,
        Y_test,
    ) = data_split(X[:, :, 0], X[:, :, 1:], Y, val=val, test=test)

    print("Done train/val/test split")
    print("Model summary:")

    # build architecture
    efn = EFN(
        input_dim=2,
        Phi_sizes=Phi_sizes,
        F_sizes=F_sizes,
        # F_dropouts=0.2,
        loss="binary_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr=lr),
        metrics=[tf.keras.metrics.AUC(name="auc")],
        output_act="sigmoid",
    )

    mc = tf.keras.callbacks.ModelCheckpoint(
        path / "models" / f"model_{layer}.h5",
        monitor="val_auc",
        verbose=1,
        save_best_only=True,
        mode="max",
    )
    es = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc", mode="max", verbose=1, patience=10
    )

    # train model
    efn.fit(
        [z_train, p_train],
        Y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=([z_val, p_val], Y_val),
        callbacks=[es, mc],
        verbose=1,
    )

    # get predictions on test data
    preds = efn.predict([z_test, p_test], batch_size=1000)

    # get ROC curve
    efn_fp, efn_tp, threshs = roc_curve(Y_test[:, 1], preds[:, 1])

    # get area under the ROC curve
    auc = roc_auc_score(Y_test[:, 1], preds[:, 1])
    print()
    print("EFN AUC:", auc)
    print()

    # some nicer plot settings
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["figure.autolayout"] = True

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ######################### ROC Curve Plot #########################

    # get multiplicity and mass for comparison
    masses = np.asarray(
        [ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X]
    )
    mults = np.asarray([np.count_nonzero(x[:, 0]) for x in X])
    mass_fp, mass_tp, threshs = roc_curve(Y[:, 1], -masses)
    mult_fp, mult_tp, threshs = roc_curve(Y[:, 1], -mults)

    # plot the ROC curves
    axes[0].plot(efn_tp, 1 - efn_fp, "-", color="black", label=f"EFN (AUC = {auc:.4})")
    axes[0].plot(mass_tp, 1 - mass_fp, "-", color="blue", label="Jet Mass")
    axes[0].plot(mult_tp, 1 - mult_fp, "-", color="red", label="Multiplicity")

    # axes labels
    axes[0].set_xlabel("Quark Jet Efficiency")
    axes[0].set_ylabel("Gluon Jet Rejection")

    # axes limits
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # make legend and show plot
    axes[0].legend(loc="lower left", frameon=False)

    ######################### Filter Plot #########################

    # plot settings
    R, n = 0.4, 100
    colors = ["Reds", "Oranges", "Greens", "Blues", "Purples", "Greys"]
    grads = np.linspace(0.45, 0.55, 4)

    # evaluate filters
    X, Y, Z = efn.eval_filters(R, n=n)

    # plot filters
    for i, z in enumerate(Z):
        axes[1].contourf(X, Y, z / np.max(z), grads, cmap=colors[i % len(colors)])

    axes[1].set_xticks(np.linspace(-R, R, 5))
    axes[1].set_yticks(np.linspace(-R, R, 5))
    axes[1].set_xticklabels(["-R", "-R/2", "0", "R/2", "R"])
    axes[1].set_yticklabels(["-R", "-R/2", "0", "R/2", "R"])
    axes[1].set_xlabel("Translated Rapidity y")
    axes[1].set_ylabel("Translated Azimuthal Angle phi")
    axes[1].set_title("Energy Flow Network Latent Space", fontdict={"fontsize": 10})

    plt.savefig(path / "figures" / f"roc_{layer}_{lr}.pdf")


if __name__ == "__main__":
    lrs = [0.001, 0.0005, 0.0001]
    layers = [96, 128]
    num_epoch = 500
    batch_size = 100

    hf = h5py.File(path.parent / "data" / "processed" / "EFN.h5", "r")
    X = hf["features"]
    y = hf["targets"]
    Y = to_categorical(y, num_classes=2)
    for layer in layers:
        for lr in lrs:
            run_EFN(X, Y, layer, lr)
