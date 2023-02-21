# Determined AI DeepSets Trial Class for hyperparameter optimisation.
# See the deepsets.py file for detailed information about the architecture.

import numpy as np

import optuna
import sklearn
import tensorflow as tf
import tensorflow.keras.layers as KL
from tensorflow import keras

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import util.util
from util.data import Data
from util.terminal_colors import tcols
from . import util as dsutil


def main(args):
    util.util.device_info()
    outdir = util.util.make_output_directory("trained_deepsets", args["outdir"])

    jet_data = Data.shuffled(**args["data_hyperparams"])

    study = optuna.create_study(
        study_name="deepsets_3layer",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        direction='maximize',
        storage=f'sqlite:///{outdir}/test.db',
        load_if_exists=True)
    study.optimize(Objective(jet_data, args), n_trials=200)

class Objective:
    def __init__(self, jet_data: Data, args: dict):
        self.jet_data = jet_data
        self.args = args
        self.training_hyperparams = {
            "epochs":      args["training_hyperparams"]["epochs"],
            "valid_split": args["training_hyperparams"]["valid_split"],
        }
        self.compilation_hyperparams = {
            "loss":     args["compilation"]["loss"],
            "metrics":  args["compilation"]["metrics"],
        }
        self.model_hyperparams = {}

    def __call__(self, trial):
        self.training_hyperparams.update(
        {
            "batch":  trial.suggest_categorical(
                "bs", self.args["training_hyperparams"]["batch"]
                ),
            "lr":     trial.suggest_float(
                "lr", *self.args["training_hyperparams"]["lr"], log=True
                ),
        })
        self.compilation_hyperparams.update({
            "optimizer": trial.suggest_categorical(
                "optim", self.args["compilation"]["optimizer"]
                ),
        })

        self.model_hyperparams.update(
        {
            "nnodes_phi": trial.suggest_categorical(
                "nphi", self.args["model_hyperparams"]["nnodes_phi"]
                ),
            "nnodes_rho": trial.suggest_categorical(
                "nrho", self.args["model_hyperparams"]["nnodes_rho"]
                ),
            "activ":      trial.suggest_categorical(
                "activ", self.args["model_hyperparams"]["activ"]
                ),
        })

        model = dsutil.choose_deepsets(
            self.args["deepsets_type"],
            self.jet_data.ncons,
            self.jet_data.nfeat,
            self.model_hyperparams,
            self.compilation_hyperparams,
            self.training_hyperparams['lr'],
        )
        model.summary(expand_nested=True)

        print(tcols.HEADER + "\n\nTRAINING THE MODEL \U0001F4AA" + tcols.ENDC)

        callbacks = get_tensorflow_callbacks()
        callbacks.append(
            optuna.integration.TFKerasPruningCallback(trial, "val_categorical_accuracy")
        )

        model.fit(
            self.jet_data.tr_data,
            self.jet_data.tr_target,
            epochs=self.training_hyperparams["epochs"],
            batch_size=self.training_hyperparams["batch"],
            verbose=2,
            callbacks=callbacks,
            validation_split=self.training_hyperparams["valid_split"],
            shuffle=True,
        )

        print(tcols.HEADER + "\nTraining done... Testing model." + tcols.ENDC)
        y_pred = tf.nn.softmax(model.predict(self.jet_data.va_data)).numpy()

        # fprs = []
        # for idx in range(y_pred.shape[1]):
            # fpr, tpr, thr = sklearn.metrics.roc_curve(y_test[:, idx], y_pred[:, idx])
            # tpr_idx = find_nearest(tpr, 0.8)
            # fprs.append(fpr[tpr_idx])

        accuracy = tf.keras.metrics.CategoricalAccuracy()
        accuracy.update_state(self.jet_data.va_target, y_pred)

        return accuracy.result().numpy()

def get_tensorflow_callbacks():
    """Prepare the callbacks for the training."""
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_categorical_accuracy", patience=20
    )
    learning = keras.callbacks.ReduceLROnPlateau(
        monitor="val_categorical_accuracy", factor=0.8, patience=10, min_lr=0.00001
    )

    return [early_stopping, learning]

class OptunaPruner(keras.callbacks.Callback):
    def __init__(self, trial):
        super(OptunaPruner, self).__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(logs["val_categorical_accuracy"], epoch)
        if self.trial.should_prune():
            trial = self.trial
            raise optuna.TrialPruned()