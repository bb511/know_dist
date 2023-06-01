import os
import numpy as np
import tensorflow as tf
import hls4ml

np.random.seed(12)
tf.random.set_seed(12)

from tensorflow import keras
import tensorflow.keras.layers as KL

from deepsets.deepsets_synth import deepsets_invariant_synth
from deepsets.deepsets_synth import deepsets_equivariant_synth


def main(args):
    util.util.device_info()
    hyperparams = util.util.load_hyperparameters_file(args["model_dir"])
    hyperparams["data_hyperparams"].update(args["data_hyperparams"])

    jet_data = Data(**hyperparams["data_hyperparams"])
    jet_data.test_data = shuffle_constituents(jet_data.test_data, args)

    model = import_model(
        hyperparams["deepsets_type"],
        data.ncons,
        data.nfeat,
        hyperparams["model_hyperparams"],
    )

    print(tcols.OKGREEN + "\n\n\nSYNTHESIZING MODEL\n" + tcols.ENDC)
    config = hls4ml.utils.config_from_keras_model(model, granularity="name")
    # config['LayerName']['dense_1']['ReuseFactor']= 2
    print(f"---------Configuration--------")
    # plotting.print_dict(config)
    hls_model = hls4ml.converters.convert_from_keras_model(
        model, hls_config=config, output_dir=outname
    )
    hls_model.compile()
    # hls_model.write()
    hls_model.build()

    # hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file=None)

def import_model(args: dict, hyperparams: dict):
    """Imports the model from a specified path. Model is saved in tf format."""
    print("Instantiating model with the hyperparameters:")
    for key in model_hyperparams:
        print(f"{key}: {model_hyperparams[key]}")

    if deepsets_type in ['sequivariant', 'sinvariant']:
        model_hyperparams.update({'input_shape': (nconst, nfeats)})

    switcher = {
        "sequivariant": lambda: deepsets_equivariant_synth(**model_hyperparams),
        "sinvariant": lambda: deepsets_invariant_synth(**model_hyperparams),
    }

    model = switcher.get(deepsets_type, lambda: None)()
    model.load_weights(os.path.join(args['model_dir'], 'model_weights.h5'))

    return model

def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print("  " * indent + str(key), end="")
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(":" + " " * (20 - len(key) - 2 * indent) + str(value))
