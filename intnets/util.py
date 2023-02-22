# Utility methods for the interaction network training, testing, etc...

import os
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras

from .qconvintnet import QConvIntNet
from .convintnet import ConvIntNet
from .densintnet import DensIntNet
from util.terminal_colors import tcols


def choose_intnet(
    intnet_type: str,
    nconst: int,
    nfeats: int,
    hyperparams: dict,
    compilation_hyperparams: dict,
    lr: float,
) -> keras.models.Model:
    """Select and instantiate a certain type of interaction network."""
    print("Instantiating model with the hyperparameters:")
    for key in hyperparams:
        print(f"{key}: {hyperparams[key]}")

    switcher = {
        "qconv": lambda: QConvIntNet(nconst, nfeats, **hyperparams),
        "dens": lambda: DensIntNet(nconst, nfeats, **hyperparams),
        "conv": lambda: ConvIntNet(nconst, nfeats, **hyperparams),
    }

    model = switcher.get(intnet_type, lambda: None)()

    comp_hps = {}
    comp_hps.update(compilation_hyperparams)
    comp_hps["optimizer"] = load_optimizer(
        comp_hps["optimizer"],
        lr,
    )
    comp_hps["loss"] = choose_loss(compilation_hyperparams["loss"])

    model.compile(**comp_hps)
    model.build((None, nconst, nfeats))
    print(tcols.OKGREEN + "Model compiled and built!" + tcols.ENDC)

    if intnet_type == "qconv":
        set_matrix_multiplication_hack_weights(model)

    if model is None:
        raise TypeError("Given interaction network model type is not implemented!")

    return model


def load_optimizer(choice: str, lr: float) -> keras.optimizers.Optimizer:
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


def set_matrix_multiplication_hack_weights(model: keras.models.Model):
    """Set the weights of the QKeras convolutional layers that are used to do
    quantised matrix multiplication to be equivalent to the elements of the first
    matrix in the multiplication. This hack to do matrix multiplication is required
    since QKeras does not have yet an implementation of matrix multiplication.
    """
    transpose_receiver = np.transpose(model._receiver_matrix)
    model.get_layer("ORr").set_weights([np.expand_dims(model._receiver_matrix, axis=0)])
    model.get_layer("ORs").set_weights([np.expand_dims(model._sender_matrix, axis=0)])
    model.get_layer("Ebar").set_weights([np.expand_dims(transpose_receiver, axis=0)])
