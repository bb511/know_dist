# Train the student network given a teacher.

import os

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow import keras

# Setting the weight initialisation seed for reproducibility.
keras.utils.set_random_seed(0)

import util.util
import util.plots
from util.data import Data
from util.terminal_colors import tcols
from . import util as stutil
from .distiller import Distiller


tf.keras.backend.set_floatx("float64")


def main(args):
    util.util.device_info()
    outdir = util.util.make_output_directory("trained_students", args["outdir"])
    util.util.save_hyperparameters_file(args, outdir)

    data = Data(**args["data_hyperparams"])

    print("Importing the teacher network model...")
    teacher = keras.models.load_model(args["teacher"], compile=False)

    print(f"Instantiating the student of type: {args['student_type']}...")
    student = stutil.choose_student(args["student_type"], args["student"])

    distiller = build_distiller(args, student, teacher)
    history = distill_knowledge(args, distiller, data)

    print(tcols.OKGREEN + "\nStudent dimensions:" + tcols.ENDC)
    student.summary(expand_nested=True)

    print(tcols.OKGREEN + "\n\n\nSAVING MODEL TO: " + tcols.ENDC, outdir)
    student.save(outdir, save_format="tf")
    plot_model_performance(history.history, outdir)


def build_distiller(args, student, teacher):
    """Instantiate and compile the knowledge distiller."""
    print("Making the distiller...")
    args["distill"]["optimizer"] = stutil.load_optimizer(
        args["distill"]["optimizer"], args["training_hyperparams"]["lr"]
    )
    distiller = Distiller(student, teacher)
    distiller.compile(**args["distill"])

    return distiller


def distill_knowledge(args, distiller, data):
    """Fit the model to the data."""
    print(tcols.HEADER + "\nTEACHING THE STUDENT \U0001F4AA" + tcols.ENDC)
    print("====================")
    stutil.print_training_attributes(args["training_hyperparams"], distiller)
    history = distiller.fit(
        data.train_data,
        data.train_target,
        epochs=args["training_hyperparams"]["epochs"],
        batch_size=args["training_hyperparams"]["batch"],
        verbose=2,
        callbacks=get_callbacks(),
        validation_split=args["training_hyperparams"]["valid_split"],
        shuffle=True,
    )

    return history


def plot_model_performance(history: dict, outdir: str):
    """Does different plots that show the performance of the trained model."""
    util.plots.loss_vs_epochs(
        outdir,
        history["student_loss"],
        history["val_student_loss"],
        "loss_epochs_student",
    )
    util.plots.accuracy_vs_epochs(
        outdir,
        history["acc"],
        history["val_acc"],
    )
    print(tcols.OKGREEN + "\nPlots done! " + tcols.ENDC)


def get_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", patience=20)
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_acc", factor=0.8, patience=10
    )

    return [early_stopping, learning]
