# Script that synthesizes a given deepsets model and then returns the performance
# performance metrics for the synthesis.
import os
import numpy as np
import tensorflow as tf
import hls4ml

np.random.seed(12)
tf.random.set_seed(12)

from tensorflow import keras
import tensorflow.keras.layers as KL

import util.util
import util.plots
import util.data
from . import flops
from util.terminal_colors import tcols


def main(args):
    util.util.device_info()
    seed = args['const_seed']
    model_dir = args['model_dir']
    plots_dir = util.util.make_output_directories(args['model_dir'], f"plots_{seed}")

    hyperparam_dict = util.util.load_hyperparameter_files(args["model_dir"])
    data = import_data(hyperparam_dict)
    data.test_data = shuffle_constituents(data.test_data, seed)
    model = import_model(model_dir, hyperparams)

    print(tcols.OKGREEN + "\n\n\nSYNTHESIZING MODEL\n" + tcols.ENDC)
    config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    config = config_hls4ml(config)

    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        hls_config=config,
        output_dir=model_dir,
        part="xcvu9p-flgb2104-2l-e",
        io_type="io_parallel",
    )
    hls_model.compile()
    report = hls_model.build(
        reset=True,
        csim=True,
        synth=True,
        cosim=False,
        validation=False,
        export=False,
        vsynth=True,
        fifo_opt=False
    )
    print(report)

    print(tcols.HEADER + f"\nRunning inference for {model_dir}" + tcols.ENDC)
    y_pred = run_inference(hls_model, data, plots_dir)
    acc_synth = calculate_accuracy(y_pred, data.test_target)
    y_pred = run_inference(model, data, plots_dir)
    acc = calculate_accuracy(y_pred, data.test_target)
    print(f"Accuracy synthed model: {acc_synth:.3f}")
    print(f"Accuracy ratio: {acc_synth/acc:.3f}")


def config_hls4ml(config: dict):
    """Adds additional configuration to hls4ml config dictionary."""
    config['Model']['Strategy']= 'Latency'

    config['LayerName']['phi1']['ParallelizationFactor']= 8
    config['LayerName']['phi1']['ReuseFactor']= 2
    config['LayerName']['phi2']['ParallelizationFactor']= 8
    config['LayerName']['phi2']['ReuseFactor']= 2
    config['LayerName']['phi3']['ParallelizationFactor']= 8
    config['LayerName']['phi3']['ReuseFactor']= 2

    for layer in config['LayerName'].keys():
        if layer == 'phi1':
            config['LayerName'][layer]['Strategy'] = "Resource"
        elif layer == 'phi2':
            config['LayerName'][layer]['Strategy'] = "Resource"
        elif layer == 'phi3':
            config['LayerName'][layer]['Strategy'] = "Resource"
        else:
            config['LayerName'][layer]['Strategy']= "Latency"

    print(f"---------Configuration--------")
    util.util.nice_print_dictionary(config)

    return config


def import_data(hyperparams):
    """Import the data used for training and validating the network."""
    fpath = hyperparams['data_hyperparams']['fpath']
    if hyperparams['data_hyperparams']['fname_test']:
        fname = hyperparams['data_hyperparams']['fname_test'].rsplit('_', 1)[0]
    else:
        fname = hyperparams['data_hyperparams']['fname']

    return util.data.Data(fpath=fpath, fname=fname, only_test=True)


def import_model(model_dir: str, hyperparams: dict):
    """Imports the model from a specified path. Model is saved in tf format."""
    print(tcols.HEADER + "Importing the model..." + tcols.ENDC)
    util.util.nice_print_dictionary("DS hps:", hyperparams["model_hyperparams"])
    model = keras.models.load_model(model_dir, compile=False)
    model.summary(expand_nested=True)

    return model

def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray):
    """Computes accuracy for a model's predictions."""
    acc = keras.metrics.CategoricalAccuracy()
    acc.update_state(y_true, y_pred)
    print(tcols.OKCYAN + f"Accuracy: " + tcols.ENDC, acc)

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


def run_inference(model: keras.Model, data: util.data.Data, plots_dir: list):
    """Computes predictions of a model and saves them to numpy files."""
    y_pred = tf.nn.softmax(model.predict(data.test_data)).numpy()
    y_pred.astype("float32").tofile(os.path.join(plots_dir, "y_pred.dat"))

    return y_pred


def calculate_accuracy(y_pred: np.ndarray, y_true: np.ndarray):
    """Computes accuracy for a model's predictions."""
    acc = keras.metrics.CategoricalAccuracy()
    acc.update_state(y_true, y_pred)
    print(tcols.OKCYAN + f"Accuracy: " + tcols.ENDC, acc)

    return acc.result().numpy()
