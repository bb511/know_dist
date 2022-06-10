# Utility methods for the knowledge distillation code.

import os

from .universal_student import UniversalStudent


def choose_intnet(args: dict) -> keras.models.Model:
    """Select and instantiate a certain type of interaction network."""
    switcher = {
        "unistudent": lambda: UniversalStudent(nconst, nfeats),
    }

    model = switcher.get(args["type"], lambda: None)()

    model.compile(**args["compilation"])
    model.build((100, nconst, nfeats))
    model.summary(expand_nested=True)

    if args["type"] == "qconv":
        set_matrix_multiplication_hack_weights(model)

    if model is None:
        raise TypeError("Given interaction network model type is not implemented!")

    return model
