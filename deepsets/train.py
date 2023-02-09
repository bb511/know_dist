# Training of the model defined in the model.py file.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(123)

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

import util.util
import util.plots
from util.data import Data
from util.terminal_colors import tcols
from . import util as dsutil

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx("float64")


def main(args):
    util.util.device_info()
    outdir = util.util.make_output_directory("trained_deepsets", args["outdir"])
    util.util.save_hyperparameters_file(args, outdir)

    data_hp = args["data_hyperparams"]
    util.util.nice_print_dictionary("DATA DEETS", data_hp)
    jet_data = Data.shuffled(**data_hp, jet_seed=args["jet_seed"], seed=args["seed"])

    model = dsutil.choose_deepsets(
        args["deepsets_type"],
        jet_data.tr_data.shape[1],
        jet_data.tr_data.shape[2],
        args["deepsets_hyperparams"],
        args["deepsets_compilation"],
    )
    model.summary(expand_nested=True)

    print(tcols.HEADER + "\n\nTRAINING THE MODEL \U0001F4AA" + tcols.ENDC)
    dsutil.print_training_attributes(model, args)
    training_hyperparams = args["training_hyperparams"]

    history = model.fit(
        jet_data.tr_data,
        jet_data.tr_target,
        epochs=training_hyperparams["epochs"],
        batch_size=training_hyperparams["batch"],
        verbose=2,
        callbacks=get_tensorflow_callbacks(),
        validation_split=training_hyperparams["valid_split"],
        shuffle=True,
    )

    model.save(outdir, save_format="tf")
    print(tcols.OKGREEN + "\nSaved model to: " + tcols.ENDC, outdir)
    plot_model_performance(history.history, outdir)


def plot_model_performance(history: dict, outdir: str):
    """Does different plots that show the performance of the trained model."""
    util.plots.loss_vs_epochs(outdir, history["loss"], history["val_loss"])
    util.plots.accuracy_vs_epochs(
        outdir,
        history["categorical_accuracy"],
        history["val_categorical_accuracy"],
    )
    print(tcols.OKGREEN + "\nPlots done! " + tcols.ENDC)


def get_tensorflow_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=20
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.8, patience=10, min_lr=0.0001
    )

    return [early_stopping, learning]
