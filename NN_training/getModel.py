# Silnence output of tensorflow/keras about GPU status
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Keras/TF imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import metrics
from keras.initializers import Orthogonal
from keras.constraints import max_norm
from keras.regularizers import l2


gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


kernel_initializer = Orthogonal(gain=1.0, seed=None)
kernel_constraint = max_norm(3)
bias_constraint = max_norm(3)


class getModel:
    def __init__(self, tp, input_shape):
        self.tp = tp
        self.input_shape = input_shape

    def LL(self):
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

    def HL(self):
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
        return model

