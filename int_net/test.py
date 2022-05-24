# Loads a model and runs it on data it was not trained on.

import os

from tensorflow import keras

from . import plots
from .data import Data
from .terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def main(args):
    seed = 123

    jet_data = Data.shuffled(
        args["data_folder"],
        args["data_hyperparams"],
        args["norm"],
        0,
        args["test_events"],
        seed=seed,
    )

    print("Importing the model...")
    model = keras.models.load_model(args["model_dir"])
    print(tcols.OKGREEN + "Model loaded! \U0001F370\U00002728\n" + tcols.ENDC)

    y_pred = model.predict(jet_data.te_data)
    plots.roc_curves(args["model_dir"], y_pred, jet_data.te_target)
    plots.dnn_output(args["model_dir"], y_pred)

    print(tcols.OKGREEN + "Done! \U0001F4C8\U00002728" + tcols.ENDC)
