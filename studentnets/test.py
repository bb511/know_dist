# Test a chosen student model on test data and check its performance.
# Make plots that quantify the performance of the student.

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

    data_hp = hyperparams["data_hyperparams"]
    util.util.nice_print_dictionary("DATA DEETS", data_hp)
    jet_data = Data.shuffled(**data_hp, jet_seed=args["jet_seed"], seed=args["seed"])

    print(tcols.HEADER + "Importing the model..." + tcols.ENDC)
    util.util.nice_print_dictionary("Student hps:", hyperparams["student"])
    model = keras.models.load_model(args["model_dir"])
    print(tcols.HEADER + "Model summary:" + tcols.ENDC)
    model.summary(expand_nested=True)
    print(tcols.OKGREEN + "Model loaded! \U0001F370\U00002728\n" + tcols.ENDC)

    print(tcols.HEADER + "Testing model:" + tcols.ENDC)

    y_pred = tf.nn.softmax(model.predict(jet_data.te_data)).numpy()
    y_pred.astype("float32").tofile(os.path.join(plots_dir, "y_pred.dat"))

    ce_loss = keras.losses.CategoricalCrossentropy()(jet_data.te_target, y_pred)
    print(tcols.OKCYAN + f"\nCross-entropy loss: {ce_loss}" + tcols.ENDC)

    util.plots.dnn_output(plots_dir, y_pred)
    util.plots.roc_curves(plots_dir, y_pred, jet_data.te_target)
    print(tcols.OKGREEN + "\nPlotted results! \U0001F4C8\U00002728" + tcols.ENDC)
