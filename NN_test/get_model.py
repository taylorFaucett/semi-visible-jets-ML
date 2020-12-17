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
from keras.initializers import Orthogonal
from keras.constraints import max_norm
from keras.regularizers import l2


kernel_initializer = Orthogonal(gain=1.0, seed=None)
kernel_constraint = max_norm(3)
bias_constraint = max_norm(3)


def LL_model(model, tp, input_shape):
    model.add(
        Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=input_shape,)
    )
    model.add(
        Conv2D(200, (4, 4), padding="same", activation="relu", input_shape=input_shape,)
    )
    model.add(MaxPooling2D((3, 3)))
    model.add(
        Conv2D(248, (1, 1), padding="same", activation="relu", input_shape=input_shape,)
    )
    model.add(
        Conv2D(192, (2, 2), padding="same", activation="relu", input_shape=input_shape,)
    )

    model.add(Flatten())

    model.add(Dense(320, activation="relu",))
    model.add(Dropout(0.3))

    model.add(Dense(96, activation="relu",))
    model.add(Dropout(0.1))

    return model


def HL_model(model, tp, input_shape):
    model.add(Flatten(input_shape=(input_shape,)))
    for dense_ix in range(int(tp["dense_layers"])):
        model.add(
            Dense(
                tp["dense_units"],
                activation="relu",
                kernel_initializer=kernel_initializer,
                kernel_constraint=kernel_constraint,
                bias_constraint=kernel_constraint,
            )
        )
        if dense_ix - 1 < int(tp["dense_layers"]):
            model.add(Dropout(tp["dropout"]))
    return model


def get_model(run_type, tp, input_shape=None):
    model = Sequential()
    if run_type == "LL":
        input_shape = (32, 32, 1)
        model = LL_model(model, tp, input_shape)

    elif run_type == "HL":
        model = HL_model(model, tp, input_shape)

    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=0.0004),
        metrics=["accuracy", metrics.AUC(name="auc")],
    )

    return model
