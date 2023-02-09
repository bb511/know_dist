# Utility methods for the interaction network training, testing, etc...

import os
import json
import numpy as np

import tensorflow as tf
from tensorflow import keras

from .terminal_colors import tcols


def make_output_directory(location: str, outdir: str) -> str:
    """Create the output directory in a designated location."""
    outdir = os.path.join(location, outdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    return outdir


def nice_print_dictionary(dictionary_name: str, dictionary: dict):
    """Logs useful details about the data used to train the interaction network."""
    print(tcols.HEADER + f"\n{dictionary_name}" + tcols.ENDC)
    print(tcols.HEADER + "-----------" + tcols.ENDC)
    if not bool(dictionary):
        print(tcols.WARNING + "Dictionary is empty.\n" + tcols.ENDC)
        return
    for key in dictionary:
        print(f"{key}: {dictionary[key]}")


def device_info():
    """Prints what device the tensorflow network will run on."""
    gpu_devices = tf.config.list_physical_devices("GPU")
    if gpu_devices:
        details = tf.config.experimental.get_device_details(gpu_devices[0])
        print(tcols.OKCYAN + f"\nGPU: {details.get('device_name')}" + tcols.ENDC)
    else:
        print(tcols.WARNING + "\nNo GPU detected. Running on CPU." + tcols.ENDC)


def save_hyperparameters_file(hyperparams: dict, outdir: str):
    """Saves the hyperparameters dictionary that defines an net to a file."""
    hyperparams_file_path = os.path.join(outdir, "hyperparameters.json")
    with open(hyperparams_file_path, "w") as file:
        json.dump(hyperparams, file)

    print(tcols.OKGREEN + "Saved hyperparameters to json file." + tcols.ENDC)


def load_hyperparameters_file(model_dir: str):
    """Loads a hyperparameters file given the directory that it's in."""
    with open(os.path.join(model_dir, "hyperparameters.json")) as file:
        hyperparams = json.load(file)

    return hyperparams


def print_training_attributes(model: keras.models.Model, args: dict):
    """Prints model attributes so all interesting infromation is printed."""
    compilation_hyperparams = args["intnet_compilation"]
    train_hyperparams = args["training_hyperparams"]

    print("\nTraining parameters")
    print("-------------------")
    print(tcols.OKGREEN + "Optimiser: \t" + tcols.ENDC, model.optimizer.get_config())
    print(tcols.OKGREEN + "Batch size: \t" + tcols.ENDC, train_hyperparams["batch"])
    print(tcols.OKGREEN + "Learning rate: \t" + tcols.ENDC, train_hyperparams["lr"])
    print(tcols.OKGREEN + "Training epochs:" + tcols.ENDC, train_hyperparams["epochs"])
    print(tcols.OKGREEN + "Loss: \t\t" + tcols.ENDC, compilation_hyperparams["loss"])
    print("")
