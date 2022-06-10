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

    jet_data = Data.shuffled(
        args["data_folder"],
        args["data_hyperparams"],
        args["norm"],
        args["train_events"],
        0,
        seed=args["seed"],
    )


    print("Importing the teacher network model...")
    teacher = keras.models.load_model(args["teacher"])
    print(tcols.OKGREEN + "Teacher loaded! \U0001f468\u200D\U0001f3eb\U00002728\n" +
          tcols.ENDC)

    print("Instantiating the student network model...")
    student = util.choose_student(args["student"])
    print(tcols.OKGREEN + "Student spawned! \U0001f468\u200D\U0001f393\U00002728\n" +
          tcols.ENDC)

    print("Making the distiller...")
    distiller = Distiller(intnet, student)
    print(tcols.OKGREEN + "Ready for knowledge transfer! \U0001F34E \n" + tcols.ENDC)

    print("Training the student...")
