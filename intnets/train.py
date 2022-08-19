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
tf.keras.backend.set_floatx('float64')

def main(args):
    util.device_info()
    outdir = util.make_output_directory("trained_intnets", args["outdir"])

    data_hp = args["data_hyperparams"]
    util.print_data_deets(data_hp)
    jet_data = Data.shuffled(**data_hp, jet_seed=args["jet_seed"], seed=args["seed"])

    nconst = jet_data.tr_data.shape[1]
    nfeats = jet_data.tr_data.shape[2]
    model = util.choose_intnet(args["inet_hyperparams"], nconst, nfeats)

    print(tcols.HEADER + "\n\nTRAINING THE MODEL \U0001F4AA" + tcols.ENDC)
    util.print_training_attributes(model, args)
    training_hyperparams = args["training_hyperparams"]

    history = model.fit(
        jet_data.tr_data,
        jet_data.tr_target,
        epochs=training_hyperparams["epochs"],
        batch_size=training_hyperparams["batch"],
        verbose=2,
        callbacks=get_tensorflow_callbacks(),
        validation_split=training_hyperparams["valid_split"],
        shuffle=False,
    )

    model.save(outdir, save_format="tf")
    print(tcols.OKGREEN + "\nSaved model to: " + tcols.ENDC, outdir)
    plot_model_performance(history.history, outdir)


def plot_model_performance(history: dict, outdir: str):
    """Does different plots that show the performance of the trained model."""
    plots.loss_vs_epochs(outdir, history["loss"], history["val_loss"])
    plots.accuracy_vs_epochs(
        outdir,
        history["categorical_accuracy"],
        history["val_categorical_accuracy"],
    )
    print(tcols.OKGREEN + "\nPlots done! \U0001f4ca" + tcols.ENDC)


def get_tensorflow_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=10
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.2, patience=10
    )

    return [early_stopping, learning]
