# Utility methods for the interaction network training, testing, etc...

import numpy as np

from tensorflow import keras

from .terminal_colors import tcols
from .qconvintnet import QConvIntNet
from .convintnet import ConvIntNet

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

def print_training_attributes(model: keras.models.Model, args: dict):
    """Prints model attributes so all interesting infromation is printed."""
    hyperparams = args["inet_hyperparams"]
    print(tcols.OKGREEN + "Optimiser: " + tcols.ENDC, model.optimizer.get_config())

    print(tcols.HEADER + "\nTRAINING PARAMETERS" + tcols.ENDC)
    print("-------------------")
    print(tcols.OKGREEN + "Batch size: \t" + tcols.ENDC, args["batch"])
    print(tcols.OKGREEN + "Training epochs:" + tcols.ENDC, args["epochs"])
    print(tcols.OKGREEN + "Loss: \t\t" + tcols.ENDC, hyperparams["compilation"]["loss"])

def choose_intnet(args: dict, nconst: int, nfeats: int) -> keras.models.Model:
    """Select and instantiate a certain type of interaction network."""
    switcher = {
        "qconv": lambda: QConvIntNet(nconst, nfeats, summation=args["summation"]),
        "conv": lambda: ConvIntNet(nconst, nfeats, summation=args["summation"]),
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
