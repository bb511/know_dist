# Training of the model defined in the model.py file.

import os

import tensorflow as tf
from tensorflow import keras

from . import util
from . import plots
from .data import Data
from .model import QConvIntNet
from .terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(args):
    tf.random.set_seed(args["seed"])
    outdir = "./trained_intnets/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    jet_data = Data.shuffled(
        args["data_folder"],
        args["data_hyperparams"],
        args["norm"],
        args["train_events"],
        0,
        seed=args["seed"],
    )

    model = QConvIntNet(jet_data.tr_data.shape[1], jet_data.tr_data.shape[2])
    model.compile(**args["inet_hyperparams"]["compilation"])
    model.build(jet_data.tr_data.shape)
    model.summary(expand_nested=True)
    util.print_model_attributes(model, args)

    print(tcols.HEADER + "\nTraining the model... \U0001F4AA" + tcols.ENDC)
    callbacks = get_callbacks()
    history = model.fit(
        jet_data.tr_data,
        jet_data.tr_target,
        epochs=args["epochs"],
        batch_size=args["batch"],
        verbose=2,
        callbacks=callbacks,
        validation_split=0.3,
    )

    print(tcols.OKGREEN + "\nSaving model to: " + tcols.ENDC, outdir)
    model.save(outdir, save_format="tf")

    plots.loss_vs_epochs(outdir, history.history["loss"], history.history["val_loss"])
    plots.accuracy_vs_epochs(
        outdir,
        history.history["categorical_accuracy"],
        history.history["val_categorical_accuracy"],
    )

    print(tcols.OKGREEN + "Done! \U0001F370\U00002728" + tcols.ENDC)


def get_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=10
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.2, patience=10
    )

    return [early_stopping, learning]
