import sys
import numpy as np
import tensorflow as tf
import hls4ml

np.random.seed(12)
tf.random.set_seed(12)

from tensorflow import keras
import tensorflow.keras.layers as KL


def main(args):
    util.util.device_info()
    outdir = util.util.make_output_directory("trained_deepsets", args["outdir"])
    util.util.save_hyperparameters_file(args, outdir)

    data = Data(**args["data_hyperparams"])

    model = build_model(args, data)
    history = train_model(model, data, args)

    print(tcols.OKGREEN + "\n\n\nSAVING MODEL TO: " + tcols.ENDC, outdir)
    model.save(outdir, save_format="tf")
    plot_model_performance(history.history, outdir)

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


def train_model(model, data, args: dict):
    """Fit the model to the data."""
    print(tcols.HEADER + "\n\nTRAINING THE MODEL \U0001F4AA" + tcols.ENDC)
    dsutil.print_training_attributes(model, args)

    history = model.fit(
        data.train_data,
        data.train_target,
        epochs=args["training_hyperparams"]["epochs"],
        batch_size=args["training_hyperparams"]["batch"],
        verbose=2,
        callbacks=get_tensorflow_callbacks(),
        validation_split=args["training_hyperparams"]["valid_split"],
        shuffle=True,
    )

    return history


def plot_model_performance(history: dict, outdir: str):
    """Does different plots that show the performance of the trained model."""
    util.plots.loss_vs_epochs(outdir, history["loss"], history["val_loss"])
    util.plots.accuracy_vs_epochs(
        outdir,
        history["categorical_accuracy"],
        history["val_categorical_accuracy"],
    )


def get_tensorflow_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=20
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.8, patience=10, min_lr=0.0001
    )

    return [early_stopping, learning]


def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print("  " * indent + str(key), end="")
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(":" + " " * (20 - len(key) - 2 * indent) + str(value))
