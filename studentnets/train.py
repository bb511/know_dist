# Train the student network given a teacher.

import os
import numpy as np

from tensorflow import keras
keras.utils.set_random_seed(123)

from . import util
from .distiller import Distiller
from intnets import util as intnet_util
from intnets.data import Data
from intnets import plots
from .terminal_colors import tcols


def main(args):

    outdir = "./trained_students/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_hyperparams = args["data_hyperparams"]
    jet_data = Data.shuffled(
        data_hyperparams["data_folder"],
        data_hyperparams["data_hyperparams"],
        data_hyperparams["norm"],
        data_hyperparams["train_events"],
        0,
        seed=args["seed"],
    )


    print("Importing the teacher network model...")
    teacher = keras.models.load_model(args["teacher"])
    print(teacher.summary())
    print(tcols.OKGREEN + "Teacher loaded! \U0001f468\u200D\U0001f3eb\U00002728\n" +
          tcols.ENDC)

    print("Instantiating the student network model...")
    student = util.choose_student(args["student"])
    print(tcols.OKGREEN + "Student spawned! \U0001f468\u200D\U0001f393\U00002728\n" +
          tcols.ENDC)

    print("Making the distiller...")
    distiller_hyperparams = args["distiller"]
    distiller = Distiller(intnet, student)
    distiller.compile(**distiller_hyperparams)
    print(tcols.OKGREEN + "Ready for knowledge transfer! \U0001F34E \n" + tcols.ENDC)

    print("Teaching the student...")
    history = distiller.fit(
        jet_data.tr_data,
        jet_data.tr_target,
        epochs=args["epochs"],
        batch_size=args["batch"],
        verbose=2,
        callbacks=get_callbacks(),
        validation_split=0.3,
    )

    print(tcols.OKGREEN + "\nSaving student model to: " + tcols.ENDC, outdir)
    student.save(outdir, save_format="tf")

    plots.loss_vs_epochs(outdir, history.history["loss"], history.history["val_loss"])
    plots.accuracy_vs_epochs(
        outdir,
        history.history["categorical_accuracy"],
        history.history["val_categorical_accuracy"],
    )

def get_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=10
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.2, patience=10
    )

    return [early_stopping, learning]
