# Training of the model defined in the model.py file.

import os
import numpy as np

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow import keras

# keras.utils.set_random_seed(123)

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

import util.util
import util.plots
from util.data import Data
from util.terminal_colors import tcols
from . import util as dsutil

tf.keras.backend.set_floatx("float64")


def main(args):
    util.util.device_info()
    outdir = util.util.make_output_directory("trained_deepsets", args["outdir"])
    util.util.save_hyperparameters_file(args, outdir)

    data = Data(**args["data_hyperparams"])

    model = build_model(args, data)
    history = train_model(model, data, args)

    print(tcols.OKGREEN + "\n\n\nSAVING MODEL TO: " + tcols.ENDC, outdir)
    model.save(outdir, save_format="tf")
    plot_model_performance(history.history, outdir)


def build_model(args: dict, data: Data):
    """Instantiate the model with chosen hyperparams and return it."""
    print(tcols.HEADER + "\n\nINSTANTIATING MODEL" + tcols.ENDC)
    model = dsutil.choose_deepsets(
        args["deepsets_type"],
        data.ncons,
        data.nfeat,
        args["model_hyperparams"],
        args["compilation"],
        args["training_hyperparams"]["lr"],
    )
    model.summary(expand_nested=True)

    return model


def train_model(model, data, args: dict):
    """Fit the model to the data."""
    print(tcols.HEADER + "\n\nTRAINING THE MODEL \U0001F4AA" + tcols.ENDC)
    dsutil.print_training_attributes(model, args)

    history = model.fit(
        data.train_data,
        data.train_target,
        epochs=args["training_hyperparams"]["epochs"],
        batch_size=args["training_hyperparams"]["batch"],
        verbose=2,
        callbacks=get_tensorflow_callbacks(),
        validation_split=args["training_hyperparams"]["valid_split"],
        shuffle=True,
    )

    return history


def plot_model_performance(history: dict, outdir: str):
    """Does different plots that show the performance of the trained model."""
    util.plots.loss_vs_epochs(outdir, history["loss"], history["val_loss"])
    util.plots.accuracy_vs_epochs(
        outdir,
        history["categorical_accuracy"],
        history["val_categorical_accuracy"],
    )


def get_tensorflow_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=20
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.8, patience=10, min_lr=0.0001
    )

    return [early_stopping, learning]
