# Silnence output of tensorflow/keras about GPU status
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import metrics
from keras.constraints import max_norm


def LL_model(model, tp):
    model.add(
        Conv2D(tp["filter_1"], (3, 3), activation="relu", input_shape=(32, 32, 1),)
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            tp["filter_2"],
            (3, 3),
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            tp["filter_3"],
            (3, 3),
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(
        Dense(
            tp["dense_units_1"],
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    model.add(Dropout(tp["dropout_1"]))
    model.add(
        Dense(
            tp["dense_units_2"],
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    model.add(Dropout(tp["dropout_2"]))
    model.add(
        Dense(
            tp["dense_units_3"],
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    return model


def HL_model(model, tp, input_shape):
    model.add(Flatten(input_shape=(input_shape,)))
    model.add(
        Dense(
            tp["dense_units_1"],
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    model.add(Dropout(tp["dropout_1"]))
    model.add(
        Dense(
            tp["dense_units_2"],
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    model.add(Dropout(tp["dropout_2"]))
    model.add(
        Dense(
            tp["dense_units_3"],
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    return model


def get_model(run_type, tp, input_shape=None):
    model = Sequential()
    if run_type == "LL":
        model = LL_model(model, tp)

    elif run_type == "HL":
        model = HL_model(model, tp, input_shape)

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=tp["learning_rate"]),
        metrics=["accuracy", metrics.AUC(name="auc")],
    )

    return model
