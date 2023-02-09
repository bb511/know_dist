# Utility methods for the interaction network training, testing, etc...

import os
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras

from deepsets.deepsets import DeepSets
from util.terminal_colors import tcols


def choose_optimiser(choice: str, lr: float) -> keras.optimizers.Optimizer:
    """Construct a keras optimiser object with a certain learning rate given a string
    for the name of that optimiser.
    """

    switcher = {
        "adam": lambda: keras.optimizers.Adam(learning_rate=lr),
    }

    optimiser = switcher.get(choice, lambda: None)()
    if optimiser is None:
        raise TypeError(
            "Optimiser was not implemented yet! Please choose one of the "
            "following: adam."
        )

    return optimiser


def choose_deepsets(
    deepsets_type: str,
    nconst: int,
    nfeats: int,
    hyperparams: dict,
    compilation_hyperparams,
) -> keras.models.Model:
    """Select and instantiate a certain type of interaction network."""
    print("Instantiating model with the hyperparameters:")
    for key in hyperparams:
        print(f"{key}: {hyperparams[key]}")

    switcher = {
        "vanilla": lambda: DeepSets(**hyperparams),
    }

    model = switcher.get(deepsets_type, lambda: None)()

    compilation_hyperparams["optimizer"] = choose_optimiser(
        compilation_hyperparams["optimizer"][0],
        compilation_hyperparams["optimizer"][1],
    )
    compilation_hyperparams["loss"] = choose_loss(compilation_hyperparams["loss"])
    model.compile(**compilation_hyperparams)
    model.build((None, nconst, nfeats))
    print(tcols.OKGREEN + "Model compiled and built!" + tcols.ENDC)

    if model is None:
        raise TypeError("Given deepsets network model type is not implemented!")

    return model


def choose_loss(choice: str, from_logits: bool = True) -> keras.losses.Loss:
    """Construct a keras optimiser object with a certain learning rate given a string
    for the name of that optimiser.
    """

    switcher = {
        "categorical_crossentropy": lambda: keras.losses.CategoricalCrossentropy(
            from_logits=from_logits
        ),
        "softmax_with_crossentropy": lambda: tf.nn.softmax_cross_entropy_with_logits,
    }

    loss = switcher.get(choice, lambda: None)()
    if loss is None:
        raise TypeError("The given loss name is not implemented in the wrapper yet!")

    return loss


def print_training_attributes(model: keras.models.Model, args: dict):
    """Prints model attributes so all interesting infromation is printed."""
    compilation_hyperparams = args["deepsets_compilation"]
    train_hyperparams = args["training_hyperparams"]

    print("\nTraining parameters")
    print("-------------------")
    print(tcols.OKGREEN + "Optimiser: \t" + tcols.ENDC, model.optimizer.get_config())
    print(tcols.OKGREEN + "Batch size: \t" + tcols.ENDC, train_hyperparams["batch"])
    print(tcols.OKGREEN + "Learning rate: \t" + tcols.ENDC, train_hyperparams["lr"])
    print(tcols.OKGREEN + "Training epochs:" + tcols.ENDC, train_hyperparams["epochs"])
    print(tcols.OKGREEN + "Loss: \t\t" + tcols.ENDC, compilation_hyperparams["loss"])
    print("")
