# Loads a model and runs it on data it was not trained on.

import os
import json

import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(123)

import util.util
import util.plots
from util.data import Data
from util.terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx("float64")


def main(args):
    util.util.device_info()
    plots_dir = util.util.make_output_directory(
        args["model_dir"], f"plots_{args['seed']}"
    )

    hyperparams = util.util.load_hyperparameters_file(args["model_dir"])
    hyperparams["data_hyperparams"].update(args["data_hyperparams"])

    jet_data = Data.shuffled(**args["data_hyperparams"])

    print(tcols.HEADER + "Importing the model..." + tcols.ENDC)
    util.util.nice_print_dictionary("DS hps:", hyperparams["model_hyperparams"])
    model = keras.models.load_model(args["model_dir"], compile=False)

    print(tcols.HEADER + "\nModel summary:" + tcols.ENDC)
    model.summary(expand_nested=True)
    print(tcols.OKGREEN + "Model loaded! \U0001F370\U00002728\n" + tcols.ENDC)

    print(tcols.HEADER + "\nPlotting" + tcols.ENDC)
    print("========")
    y_pred = tf.nn.softmax(model.predict(jet_data.te_data)).numpy()
    print(tcols.OKGREEN + "\nSaved predictions array.\n" + tcols.ENDC)
    y_pred.astype("float32").tofile(os.path.join(plots_dir, "y_pred.dat"))

    ce_loss = keras.losses.CategoricalCrossentropy()(jet_data.te_target, y_pred)
    print(tcols.OKCYAN + f"\nCross-entropy loss: {ce_loss}" + tcols.ENDC)

    util.plots.roc_curves(plots_dir, y_pred, jet_data.te_target)
    util.plots.dnn_output(plots_dir, y_pred)
    print(tcols.OKGREEN + "\nPlotting done! \U0001F4C8\U00002728" + tcols.ENDC)
