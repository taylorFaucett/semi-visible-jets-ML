from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split
import pandas as pd
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics, preprocessing
import os
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

import pathlib
path = pathlib.Path.cwd()

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
    f1 = path.parent / "data" / "efp" / rinv / "2_1_0_k_1_b_1.feather"
    f2 = path.parent / "data" / "efp" / rinv / "2_3_0_k_2_b_2.feather"
    x1 = pd.read_feather(f1)
    x2 = pd.read_feather(f2)
    y = x1["targets"].values
    d1 = x1["features"]
    d2 = x2["features"]
    x = pd.concat([d1, d2], axis=1)
    if N is not None:
        x = x.loc[:N-1]
        y = y[:N]
    scaler = preprocessing.StandardScaler()
    x = scaler.fit_transform(x)
    return x, y

def plot_roc(X_test, y_test, rinv):
    auc_save_file = path / f"auc-{rinv}.txt"
    test_predictions = model.predict(X_test).ravel()
    auc = metrics.roc_auc_score(y_test, test_predictions)    
    with open(auc_save_file, 'w') as f:
        f.write(str(auc))

    fpr, tpr, thresholds = metrics.roc_curve(y_test, test_predictions)
    background_efficiency = fpr
    signal_efficiency = tpr
    # background_rejection = 1. - background_efficiency
    background_rejection = 1./fpr
    rinv_str = rinv.replace("p", ".")
    plt.plot(signal_efficiency, background_rejection,
         lw=2, label='$r_{inv} = %s$ ($AUC = %0.3f$)' %(rinv_str, auc))
    plt.yscale("log")
    plt.xlabel('Signal efficiency $(\epsilon_S)$')
    plt.ylabel('Background rejection $(1 / \epsilon_B)$')
    plt.xlim([0,1])
    plt.title('ROC - CNN on Jet Images')
    plt.legend(loc="upper right")
    plt.savefig(path / "figures" / "cnn_roc.png")
    plt.savefig(path / "figures" / "cnn_roc.pdf")
    return auc

def train_dnn(X, y, rinv, retrain=False):
    # To retrain, remove the old model
    model_file = path / "models" / f"{rinv}.h5"
    if retrain and model_file.exists():
        os.remove(model_file)
        
    
    # Split data for train, test, validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_val, y_val, test_size=0.5, random_state=42
    )

    # If the model already exists (i.e. we haven't removed the last one) train a new model
    if model_file.exists():
        return tf.keras.models.load_model(model_file), X_test, y_test
    else:
        
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.01,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            amsgrad=False,
            name="Adam",
        )
        
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(X_train.shape[1],)))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(64, activation='relu'))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.compile(
            loss="binary_crossentropy",
            optimizer=optimizer,
            metrics=[
                tf.keras.metrics.AUC(name="auc"),
                tf.keras.metrics.Accuracy(name="acc"),
            ],
        )
        mc = tf.keras.callbacks.ModelCheckpoint(
            model_file,
            verbose=2,
            save_best_only=True,
        )
        es = tf.keras.callbacks.EarlyStopping(verbose=2, patience=10)

        history = model.fit(X_train, 
                            y_train, 
                            batch_size = 128, 
                            epochs=250,
                            verbose=2,
                            validation_data=(X_val, y_val),
                            callbacks=[mc, es],
                           )
        
        return model, X_test, y_test

    
    
if __name__ == "__main__":
    rinvs = ["0p0", "0p3", "1p0"]
    for rinv in rinvs:
        # Grab jet images and labels
        X, y = get_data(rinv, N=20000)

        # Train a new model (or load the existing one if available)
        model, X_test, y_test = train_dnn(X, y, rinv, retrain=True)

        # Plot the ROC curve
        auc_val = plot_roc(X_test, y_test, rinv)
        print(rinv, auc_val)

        # Generate predictions for the full dataset
        full_predictions = np.concatenate(model.predict(X))

        # Save the predictions
        np.save(path / "predictions" / "ll_predictions.npy", full_predictions)