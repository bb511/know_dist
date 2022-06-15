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

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(args):
    intnet_util.device_info()
    outdir = "./trained_students/" + args["outdir"] + "/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_hyperparams = args["data_hyperparams"]
    jet_data = Data.shuffled(**data_hyperparams, seed=args["seed"])

    print("Importing the teacher network model...")
    teacher = keras.models.load_model(args["teacher"])
    print(
        tcols.OKGREEN
        + "Teacher loaded! \U0001f468\u200D\U0001f3eb\U00002728\n"
        + tcols.ENDC
    )

    print("Instantiating the student network model...")
    student = util.choose_student(args["student"], jet_data.tr_data.shape)
    print(
        tcols.OKGREEN
        + "Student spawned! \U0001f468\u200D\U0001f393\U00002728\n"
        + tcols.ENDC
    )

    print("Making the distiller...")
    distiller_hyperparams = args["distill"]
    distiller = Distiller(teacher, student)
    distiller.compile(**distiller_hyperparams)
    print(tcols.OKGREEN + "Ready for knowledge transfer! \U0001F34E \n" + tcols.ENDC)

    print(tcols.HEADER + "\nTEACHING THE STUDENT \U0001F4AA" + tcols.ENDC)
    print("====================")
    training_hyperparams = args["training_hyperparams"]
    util.print_training_attributes(training_hyperparams, distiller_hyperparams)
    history = distiller.fit(
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
    print(tcols.OKGREEN + "Saving student model to: " + tcols.ENDC, outdir)
    student.save(outdir, save_format="tf")

    plots.loss_vs_epochs(
        outdir, history.history["student_loss"], history.history["val_student_loss"]
    )
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
