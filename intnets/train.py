# Training of the model defined in the model.py file.

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(123)

import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)

from . import util
from . import plots
from .data import Data
from .terminal_colors import tcols

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.keras.backend.set_floatx('float64')

def main(args):
    util.device_info()

    outdir = os.path.join("trained_intnets", args["outdir"])
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    data_hyperparams = args["data_hyperparams"]
    jet_data = Data.shuffled(
        **data_hyperparams, jet_seed=args["jet_seed"], seed=args["seed"]
    )

    # jet_data_1 = Data.shuffled(
    #     **data_hyperparams, jet_seed=args["jet_seed"], seed=273
    # )

    nconst = jet_data.tr_data.shape[1]
    nfeats = jet_data.tr_data.shape[2]

    print(tcols.OKGREEN + f"Number of constituents: {nconst}" + tcols.ENDC)

    model = util.choose_intnet(args["inet_hyperparams"], nconst, nfeats)
    # model_1 = util.choose_intnet(args["inet_hyperparams"], nconst, nfeats)
    # model_1.set_weights(model.get_weights())

    print(tcols.HEADER + "\nTRAINING THE MODEL \U0001F4AA" + tcols.ENDC)
    print("==================")
    util.print_training_attributes(model, args)
    training_hyperparams = args["training_hyperparams"]

    # optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    # optimizer_1 = keras.optimizers.Adam(learning_rate=0.0005)

    # loss = keras.losses.CategoricalCrossentropy()
    # loss_1 = keras.losses.CategoricalCrossentropy()

    # train_dataset = tf.data.Dataset.from_tensor_slices(
    #     (jet_data.tr_data, jet_data.tr_target)
    # )
    # train_dataset = train_dataset.batch(
    #     training_hyperparams["batch"]
    # )

    # train_dataset_1 = tf.data.Dataset.from_tensor_slices(
    #     (jet_data_1.tr_data, jet_data_1.tr_target)
    # )
    # train_dataset_1 = train_dataset_1.batch(
    #     training_hyperparams["batch"]
    # )

    # for epoch in range(training_hyperparams["epochs"]):
    #     print(f"\nStart of epoch {epoch}")
    #     for (x_batch_train, y_batch_train), (x_batch_train_1, y_batch_train_1) in zip(train_dataset, train_dataset_1):

    #         with tf.GradientTape() as tape:
    #             logits = model(x_batch_train, training=True)
    #             loss_value = loss(y_batch_train, logits)

    #         with tf.GradientTape() as tape_1:
    #             logits_1 = model_1(x_batch_train_1, training=True)
    #             loss_value_1 = loss_1(y_batch_train_1, logits_1)

    #         grads = tape.gradient(loss_value, model.trainable_weights)
    #         grads_1 = tape_1.gradient(loss_value_1, model_1.trainable_weights)
            # for idx in range(len(grads)):
                # print(tf.reduce_all(tf.equal(grads[idx], grads_1[idx])))
            # print(grads[-1])
            # print(grads_1[-1])
            # print()

            # optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # optimizer_1.apply_gradients(zip(grads_1, model_1.trainable_weights))

    history = model.fit(
        jet_data.tr_data,
        jet_data.tr_target,
        epochs=training_hyperparams["epochs"],
        batch_size=training_hyperparams["batch"],
        verbose=2,
        callbacks=get_callbacks(),
        validation_split=0.3,
        shuffle=False
    )

    print(tcols.HEADER + "\nSAVING RESULTS" + tcols.ENDC)
    print("==============")
    model.save(outdir, save_format="tf")
    print(tcols.OKGREEN + "\nSaved model to: " + tcols.ENDC, outdir)
    plots.loss_vs_epochs(outdir, history.history["loss"], history.history["val_loss"])
    plots.accuracy_vs_epochs(
        outdir,
        history.history["categorical_accuracy"],
        history.history["val_categorical_accuracy"],
    )

    print(tcols.OKGREEN + "\nJobs done! \U0001F370\U00002728" + tcols.ENDC)


def get_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=10
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.2, patience=10
    )

    return [early_stopping, learning]
