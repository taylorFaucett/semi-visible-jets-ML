# Silnence output of tensorflow/keras about GPU status
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import metrics

def cnn_model(tp):
    model = Sequential()
    model.add(Conv2D(tp["filter_1"], (tp["kernel_1"], tp["kernel_1"]), padding="same", activation='relu', input_shape=(32, 32, 1)))
    model.add(MaxPooling2D((tp["pool_1"], tp["pool_1"])))
    model.add(Conv2D(tp["filter_2"], (tp["kernel_2"], tp["kernel_2"]), padding="same", activation='relu'))
    model.add(Conv2D(tp["filter_3"], (tp["kernel_3"], tp["kernel_3"]), padding="same", activation='relu'))
    model.add(Flatten())
    model.add(Dense(tp["dense_units_1"], activation='relu'))
    model.add(Dropout(tp["dropout_1"]))
    model.add(Dense(tp["dense_units_2"], activation='relu'))
    model.add(Dropout(tp["dropout_2"]))
    model.add(Dense(tp["dense_units_3"], activation='relu'))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=tp['learning_rate']),
        metrics=["accuracy", metrics.AUC(name="auc")],
    )
    return model