import glob
import os

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


def nn(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    layers,
    nodes,
    model_file,
    verbose,
):

    if os.path.isfile(model_file):
        print("Using existing model")
        print(model_file)
        return keras.models.load_model(model_file)
    else:
        X_train = X_train.to_numpy()
        X_val = X_val.to_numpy()
        # print("    Training a new model at: " + model_file)
        model = keras.Sequential()
        model.add(Dense(X_train.shape[-1], activation="relu"))
        for lix in range(layers):
            model.add(Dense(nodes, activation="relu"))
            # if lix <= layers - 2:
            #     model.add(Dropout(0.25))
        model.add(Dense(1, activation="sigmoid"))
        model.compile(
            loss="binary_crossentropy",
            optimizer=keras.optimizers.Adam(),
            metrics=["accuracy"],
        )

        mc = keras.callbacks.ModelCheckpoint(
            str(model_file), verbose=verbose, save_best_only=True,
        )
        es = keras.callbacks.EarlyStopping(patience=10)

        model.fit(
            X_train,
            y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_val, y_val),
            callbacks=[mc, es],
        )

        return model
