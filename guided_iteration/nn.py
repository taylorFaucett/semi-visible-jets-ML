import glob
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices("GPU")
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
    model_file,
    verbose,
):

    if os.path.isfile(model_file):
        print("Using existing model")
        print(model_file)
        return tf.keras.models.load_model(model_file)
    else:
        # print("    Training a new model at: " + model_file)
        model = tf.keras.Sequential()

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=0.0001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07,
            amsgrad=False,
            name="Adam",
        )

        model.add(tf.keras.layers.Flatten(input_dim=X_train.shape[1]))
        for lix in range(layers):
            model.add(
                tf.keras.layers.Dense(
                    nodes,
                    kernel_initializer="normal",  # tf.keras.initializers.GlorotNormal(),
                    activation="relu",
                    kernel_constraint=tf.keras.constraints.MaxNorm(3),
                    bias_constraint=tf.keras.constraints.MaxNorm(3),
                )
            )
        #             if lix <= layers - 2:
        #                 model.add(tf.keras.layers.Dropout(0.25))

        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
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
            monitor="val_auc",
            verbose=verbose,
            save_best_only=True,
            mode="max",
        )
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max", verbose=verbose, patience=10
        )

        if verbose > 0:
            print(model.summary())

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
