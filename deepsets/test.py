# Loads a model and runs it on data it was not trained on.

import os
import json
import glob

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn

# keras.utils.set_random_seed(123)

import util.util
import util.plots
import util.data
from . import flops
from util.terminal_colors import tcols

# Set keras float precision. Default is float32.
# tf.keras.backend.set_floatx("float64")


def main(args):
    util.util.device_info()
    if "kfold" in args["model_dir"]:
        kfold_root_dir = args["model_dir"]
        args["model_dir"] = glob.glob(os.path.join(args["model_dir"], "*kfold*"))
    plots_dir = util.util.make_output_directories(
        args["model_dir"], f"plots_{args['const_seed']}"
    )
    if isinstance(args["model_dir"], str):
        args["model_dir"] = [args["model_dir"]]

    hyperparam_dicts = util.util.load_hyperparameter_files(args["model_dir"])
    kfold_fprs = []
    kfold_aucs = []
    kfold_fats = []
    kfold_accs = []
    kfold_loss = []

    for idx, hyperparams in enumerate(hyperparam_dicts):
        model_dir = args["model_dir"][idx]
        data = import_data(hyperparams)
        # data.test_data = shuffle_constituents(data.test_data, args['const_seed'])

        model = import_model(model_dir, hyperparams)
        if hyperparams["deepsets_type"] in ["equivariant", "invariant"]:
            count_flops(model_dir, model)

        print(tcols.HEADER + f"\nRunning inference and saving preds" + tcols.ENDC)
        y_pred = tf.nn.softmax(model.predict(data.test_data)).numpy()
        y_pred.astype("float32").tofile(os.path.join(plots_dir[idx], "y_pred.dat"))

        ce_loss = keras.losses.CategoricalCrossentropy()(
            data.test_target, y_pred
        ).numpy()
        print(tcols.OKCYAN + f"Cross-entropy loss: " + tcols.ENDC, ce_loss)
        kfold_loss.append(ce_loss)

        acc = calculate_accuracy(y_pred, data.test_target)
        print(tcols.OKCYAN + f"Accuracy: " + tcols.ENDC, acc)
        kfold_accs.append(acc)

        fprs, tpr_baseline, aucs, fats = util.plots.roc_curves(
            plots_dir[idx], y_pred, data.test_target
        )
        util.plots.dnn_output(plots_dir[idx], y_pred)
        print(tcols.OKGREEN + "\nPlotting done! \U0001F4C8\U00002728" + tcols.ENDC)
        kfold_fprs.append(fprs)
        kfold_aucs.append(aucs)
        kfold_fats.append(fats)

        print(tcols.OKGREEN + "\nSaving the model weights..." + tcols.ENDC)
        weights_file_path = os.path.join(model_dir, "model_weights.h5")
        model.save_weights(weights_file_path, save_format="h5")

    if len(kfold_fprs) > 1:
        print(tcols.HEADER + "\nAVERAGE RESULTS..." + tcols.ENDC)
        outdir = util.util.make_output_directory(kfold_root_dir, "average_plots")
        average_fprs = np.mean(kfold_fprs, axis=0)
        errors_fprs = np.std(kfold_fprs, axis=0)
        average_aucs = np.mean(kfold_aucs, axis=0)
        errors_aucs = np.std(kfold_aucs, axis=0)
        average_fats = np.mean(kfold_fats, axis=0)
        errors_fats = np.std(kfold_fats, axis=0)
        variance_fats = np.var(kfold_fats, axis=0)
        util.plots.roc_curves_uncert(
            tpr_baseline,
            average_fprs,
            errors_fprs,
            average_aucs,
            errors_aucs,
            average_fats,
            errors_fats,
            outdir,
        )

        print(
            f"Accuracy breakdown: {np.mean(kfold_accs, axis=0):.3f} \u00B1 {np.std(kfold_accs):.3f}"
        )
        print(
            f"Average loss: {np.mean(kfold_loss, axis=0):.3f} \u00B1 {np.std(kfold_loss):.3f}"
        )
        inverse_fats = 1 / np.mean(average_fats)
        inverse_fats_error = inverse_fats**2 * np.sqrt(np.sum(variance_fats))
        print(f"Average 1/<FPR>: {inverse_fats:.3f} \u00B1 {inverse_fats_error:.3f}")


def import_data(hyperparams):
    """Import the data used for training and validating the network."""
    if "fnames_train" in hyperparams["data_hyperparams"].keys():
        return util.data.Data.load_kfolds(**hyperparams["data_hyperparams"])

    return util.data.Data(**hyperparams["data_hyperparams"])


def import_model(model_dir: str, hyperparams: dict):
    """Imports the model from a specified path. Model is saved in tf format."""
    print(tcols.HEADER + "Importing the model..." + tcols.ENDC)
    util.util.nice_print_dictionary("DS hps:", hyperparams["model_hyperparams"])
    model = keras.models.load_model(model_dir, compile=False)
    model.summary(expand_nested=True)

    return model


def count_flops(model_dir: str, model: keras.Model):
    nflops = flops.get_flops(model)
    print("\n".join(f"{k} FLOPs: {v}" for k, v in nflops.items()))
    print(f"{'':=<65}")
    util.util.save_flops_file(nflops, model_dir)


def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray):
    """Computes accuracy for a model's predictions."""
    acc = keras.metrics.CategoricalAccuracy()
    acc.update_state(y_true, y_pred)

    return acc.result().numpy()


def shuffle_constituents(data: np.ndarray, const_seed: int) -> np.ndarray:
    """Shuffles the constituents based on an array of seeds.

    Note that each jet's constituents is shuffled with respect to a seed that is fixed.
    This seed is different for each jet.
    """
    print("Shuffling constituents...")

    rng = np.random.default_rng(const_seed)
    seeds = rng.integers(low=0, high=10000, size=data.shape[0])

    for jet_idx, seed in enumerate(seeds):
        shuffling = np.random.RandomState(seed=seed).permutation(data.shape[1])
        data[jet_idx, :] = data[jet_idx, shuffling]

    print(tcols.OKGREEN + f"Shuffling done! \U0001F0CF" + tcols.ENDC)

    return data
