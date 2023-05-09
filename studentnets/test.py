# Test a chosen student model on test data and check its performance.
# Make plots that quantify the performance of the student.

import os
import json
import numpy as np

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from tensorflow import keras
# keras.utils.set_random_seed(0)

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
        args["model_dir"], f"plots_{args['const_seed']}"
    )

    hyperparams = util.util.load_hyperparameters_file(args["model_dir"])
    hyperparams["data_hyperparams"].update(args["data_hyperparams"])

    jet_data = Data(**hyperparams["data_hyperparams"])

    print("Shuffling constituents...")
    rng = np.random.default_rng(args["const_seed"])
    seeds = rng.integers(low=0, high=10000, size=jet_data.ntest_jets)
    jet_data.test_data = shuffle_constituents(jet_data.test_data, seeds)
    print(tcols.OKGREEN + f"Shuffling done! \U0001F0CF" + tcols.ENDC)

    print(tcols.HEADER + "Importing the model..." + tcols.ENDC)
    util.util.nice_print_dictionary("Student hps:", hyperparams["student"])
    model = keras.models.load_model(args["model_dir"])
    print(tcols.HEADER + "Model summary:" + tcols.ENDC)
    model.summary(expand_nested=True)
    print(tcols.OKGREEN + "Model loaded! \U0001F370\U00002728\n" + tcols.ENDC)

    print(tcols.HEADER + "Testing model:" + tcols.ENDC)

    y_pred = tf.nn.softmax(model.predict(jet_data.test_data)).numpy()
    y_pred.astype("float32").tofile(os.path.join(plots_dir, "y_pred.dat"))

    ce_loss = keras.losses.CategoricalCrossentropy()(jet_data.test_target, y_pred)
    print(tcols.OKCYAN + f"\nCross-entropy loss: {ce_loss}" + tcols.ENDC)

    util.plots.dnn_output(plots_dir, y_pred)
    util.plots.roc_curves(plots_dir, y_pred, jet_data.test_target)
    print(tcols.OKGREEN + "\nPlotted results! \U0001F4C8\U00002728" + tcols.ENDC)

def shuffle_constituents(data: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Shuffles the constituents based on an array of seeds.

    Note that each jet's constituents is shuffled with respect to a seed that is fixed.
    This seed is different for each jet.
    """

    for jet_idx, seed in enumerate(seeds):
        shuffling = np.random.RandomState(seed=seed).permutation(data.shape[1])
        data[jet_idx, :] = data[jet_idx, shuffling]

    return data
