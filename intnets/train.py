# Training of the model defined in the model.py file.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(123)

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

from . import util
from . import plots
from .data import Data
from .terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(args):
    util.device_info()

    outdir = os.path.join("trained_intnets", args["outdir"])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_hyperparams = args["data_hyperparams"]
    jet_data = Data.shuffled(
        **data_hyperparams, jet_seed=args["jet_seed"], seed=args["seed"]
    )

    nconst = jet_data.tr_data.shape[1]
    nfeats = jet_data.tr_data.shape[2]

    print(tcols.OKGREEN + f"Number of constituents: {nconst}" + tcols.ENDC)

    model = util.choose_intnet(args["inet_hyperparams"], nconst, nfeats)

    print(tcols.HEADER + "\nTRAINING THE MODEL \U0001F4AA" + tcols.ENDC)
    print("==================")
    # util.print_training_attributes(model, args)
    training_hyperparams = args["training_hyperparams"]

    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    loss = keras.losses.CategoricalCrossentropy()

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (jet_data.tr_data, jet_data.tr_target)
    )
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(
        training_hyperparams["batch"]
    )

    # for epoch in range(training_hyperparams["epochs"]):
    #     print(f"\nStart of epoch {epoch}")
    #     for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    #         with tf.GradientTape() as tape:
    #             logits = model(x_batch_train, training=True)
    #             loss_value = loss(y_batch_train, logits)

    #         grads = tape.gradient(loss_value, model.trainable_weights)
    #         print(grads[1])
    #         optimizer.apply_gradients(zip(grads, model.trainable_weights))

    #         print(f"Training loss at step {step}: {loss_value:.2f}")

    history = model.fit(
        jet_data.tr_data,
        jet_data.tr_target,
        epochs=training_hyperparams["epochs"],
        batch_size=training_hyperparams["batch"],
        verbose=2,
        callbacks=get_callbacks(),
        validation_split=0.3,
    )

    print(tcols.HEADER + "\nSAVING RESULTS" + tcols.ENDC)
    print("==============")
    model.save(outdir, save_format="tf")
    print(tcols.OKGREEN + "\nSaved model to: " + tcols.ENDC, outdir)
    plots.loss_vs_epochs(outdir, history.history["loss"], history.history["val_loss"])
    plots.accuracy_vs_epochs(
        outdir,
        history.history["categorical_accuracy"],
        history.history["val_categorical_accuracy"],
    )

    print(tcols.OKGREEN + "\nJobs done! \U0001F370\U00002728" + tcols.ENDC)


def get_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=10
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.2, patience=10
    )

    return [early_stopping, learning]
