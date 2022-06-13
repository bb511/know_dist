# Utility methods for the knowledge distillation code.

import os

from tensorflow import keras
from .universal_student import UniversalStudent
from .terminal_colors import tcols


def choose_student(args: dict, data_shape: tuple[int]) -> keras.models.Model:
    """Select and instantiate a certain type of interaction network."""
    switcher = {
        "unistudent": lambda: UniversalStudent(),
    }

    model = switcher.get(args["type"], lambda: None)()

    if model is None:
        raise TypeError("Given interaction network model type is not implemented!")

    print(tcols.WARNING + "Cannot display structure of student since compilation "
          "happens in the distiller class." + tcols.ENDC)
    return model

def choose_loss(choice: str, from_logits: bool) -> keras.losses.Loss:
    """Construct a keras optimiser object with a certain learning rate given a string
    for the name of that optimiser.
    """

    switcher = {
        "categorical_crossentropy":
            lambda: keras.losses.CategoricalCrossentropy(from_logits=from_logits),
        "kl_divergence": lambda: keras.losses.KLDivergence(),
    }

    loss = switcher.get(choice, lambda: None)()
    if loss is None:
        raise TypeError("The given loss name is not implemented in the wrapper yet!")

    return loss

def print_training_attributes(train_hp: dict, dist_hp: dict):
    """Prints model attributes so all interesting infromation is printed."""
    print(tcols.OKGREEN + "Distillation optimiser: " + tcols.ENDC, dist_hp["optimizer"])

    print(tcols.HEADER + "\nTRAINING PARAMETERS" + tcols.ENDC)
    print("-------------------")
    print(tcols.OKGREEN + "Batch size: \t" + tcols.ENDC, train_hp["batch"])
    print(tcols.OKGREEN + "Training epochs:" + tcols.ENDC, train_hp["epochs"])
    print(tcols.OKGREEN + "Loss student: \t\t" + tcols.ENDC, dist_hp["student_loss_fn"])
    print(tcols.OKGREEN + "Loss distill: \t\t" + tcols.ENDC, dist_hp["distill_loss_fn"])

