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


def get_model(tp):
    model = Sequential()
    model.add(
        Conv2D(
            tp["filter_1"],
            (3, 3),
            padding="valid",
            activation="relu",
            input_shape=(32, 32, 1),
        )
    )
    model.add(MaxPooling2D((2, 2)))
    model.add(
        Conv2D(
            tp["filter_2"],
            (3, 3),
            padding="valid",
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
            padding="valid",
            activation="relu",
            kernel_constraint=max_norm(3),
            bias_constraint=max_norm(3),
        )
    )
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dropout(tp["dropout_0"]))
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
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=tp["learning_rate"]),
        metrics=["accuracy", metrics.AUC(name="auc")],
    )
    
    return model
