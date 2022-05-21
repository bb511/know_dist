# Loads a model and runs it on data it was not trained on.

import os

from tensorflow import keras

from .data import Data
from .terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(args):
    seed = 123

    jet_data = Data(
        args["data_folder"],
        args["data_hyperparams"],
        args["norm"],
        0,
        args["test_events"],
        seed=seed,
    )

    print("Importing the model...")
    model = keras.models.load_model(args["model_dir"])
    print(tcols.OKGREEN + "Model loaded! \U0001F370\U00002728" + tcols.ENDC)
