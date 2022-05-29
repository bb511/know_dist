# Utility methods for the interaction network training, testing, etc...

from tensorflow import keras

from .terminal_colors import tcols


def choose_optimiser(choice: str, lr: float) -> keras.optimizers:
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


def print_model_attributes(model: keras.models.Model, args: dict):
    """Prints model attributes so all interesting infromation is printed."""
    hyperparams = args["inet_hyperparams"]
    print(tcols.OKGREEN + "Optimiser: " + tcols.ENDC, model.optimizer.get_config())

    print(tcols.HEADER + "\nTRAINING PARAMETERS" + tcols.ENDC)
    print("-------------------")
    print(tcols.OKGREEN + "Batch size: \t" + tcols.ENDC, args["batch"])
    print(tcols.OKGREEN + "Training epochs:" + tcols.ENDC, args["epochs"])
    print(tcols.OKGREEN + "Loss: \t\t" + tcols.ENDC, hyperparams["compilation"]["loss"])
