# Silnence output of tensorflow/keras about GPU status
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Keras imports
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import metrics

def HL_model(tp, input_shape):    
    model = Sequential()
    model.add(Flatten(input_shape=(input_shape,)))
    model.add(Dense(tp["dense_units_1"], activation="relu"))
    model.add(Dropout(tp["dropout_1"]))
    model.add(Dense(tp["dense_units_2"], activation="relu"))
    model.add(Dropout(tp["dropout_2"]))
    model.add(Dense(tp["dense_units_3"], activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(
        loss="binary_crossentropy",
        optimizer=Adam(lr=tp['learning_rate']),
        metrics=["accuracy", metrics.AUC(name="auc")],
    )
    return model