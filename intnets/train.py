# Training of the model defined in the model.py file.

import os
import numpy as np

from tensorflow import keras
keras.utils.set_random_seed(123)

from . import util
from . import plots
from .data import Data
from .terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(args):
    outdir = "./trained_intnets/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_hyperparams = args["data_hyperparams"]
    jet_data = Data.shuffled(data_hyperparams, seed=args["seed"])

    nconst = jet_data.tr_data.shape[1]
    nfeats = jet_data.tr_data.shape[2]

    model = util.choose_intnet(args["inet_hyperparams"], nconst, nfeats)

    print(tcols.HEADER + "\nTraining the model... \U0001F4AA" + tcols.ENDC)
    util.print_training_attributes(model, args)
    training_hyperparams = args["training_hyperparams"]
    history = model.fit(
        jet_data.tr_data,
        jet_data.tr_target,
        epochs=training_hyperparams["epochs"],
        batch_size=training_hyperparams["batch"],
        verbose=2,
        callbacks=get_callbacks(),
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
