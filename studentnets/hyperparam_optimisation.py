# Optuna DeepSets Trial Class for hyperparameter optimisation.
# See the deepsets.py file for detailed information about the architecture.

import os
import numpy as np

# Silence the info from tensorflow in which it brags that it can run on cpu nicely.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
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
from . import util as stutil
from .distiller import Distiller


def main(args):
    util.util.device_info()
    outdir = util.util.make_output_directory("trained_students", args["outdir"])

    jet_data = Data(**args["data_hyperparams"])

    study = optuna.create_study(
        study_name="3layer_extremely_dumb_student",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.SuccessiveHalvingPruner(),
        direction="maximize",
        storage=f"sqlite:///{outdir}/deepsets_deepsets_equiv.db",
        load_if_exists=True,
    )
    study.optimize(Objective(jet_data, args), n_trials=150, gc_after_trial=True)


class Objective:
    def __init__(self, jet_data: Data, args: dict):
        self.jet_data = jet_data
        self.args = args
        self.training_hyperparams = {
            "epochs": args["training_hyperparams"]["epochs"],
            "valid_split": args["training_hyperparams"]["valid_split"],
        }
        self.distillation_params = {
            "student_loss_fn": args["distill"]["student_loss_fn"],
            "distill_loss_fn": args["distill"]["distill_loss_fn"],
        }

    def __call__(self, trial):
        tf.keras.backend.clear_session()
        self.training_hyperparams.update(
            {
                "batch": trial.suggest_categorical(
                    "bs", self.args["training_hyperparams"]["batch"]
                ),
                "lr": trial.suggest_float(
                    "lr", *self.args["training_hyperparams"]["lr"], log=True
                ),
            }
        )
        self.distillation_params.update(
            {
                "optimizer": trial.suggest_categorical(
                    "optim", self.args["distill"]["optimizer"]
                ),
                "alpha": trial.suggest_float(
                    "alpha",
                    self.args['distill']['alpha'][0],
                    self.args['distill']['alpha'][1],
                    step=self.args['distill']['alpha'][2],
                ),
                "temperature": trial.suggest_int(
                    "temperature", *self.args['distill']['temperature'],
                ),
            }
        )

        teacher = keras.models.load_model(self.args['teacher'], compile=False)
        student = stutil.choose_student(self.args['student_type'], self.args['student'])

        distiller = self.build_distiller(student, teacher)

        print(tcols.HEADER + "\n\nTRAINING THE MODEL \U0001F4AA" + tcols.ENDC)

        callbacks = self.get_tensorflow_callbacks()
        callbacks.append(
            optuna.integration.TFKerasPruningCallback(trial, "val_acc")
        )

        stutil.print_training_attributes(self.args["training_hyperparams"], distiller)
        distiller.fit(
            self.jet_data.train_data,
            self.jet_data.train_target,
            epochs=self.training_hyperparams["epochs"],
            batch_size=self.training_hyperparams["batch"],
            verbose=2,
            callbacks=callbacks,
            validation_split=self.training_hyperparams["valid_split"],
            shuffle=True,
        )

        print(tcols.HEADER + "\nTraining done... Testing model." + tcols.ENDC)
        student = distiller.student
        y_pred = tf.nn.softmax(student.predict(self.jet_data.valid_data)).numpy()

        accuracy = tf.keras.metrics.CategoricalAccuracy()
        accuracy.update_state(self.jet_data.valid_target, y_pred)

        return accuracy.result().numpy()

    def build_distiller(self, student, teacher):
        """Instantiate and compile the knowledge distiller."""
        print("Making the distiller...")
        distillation_params = {}
        distillation_params.update(self.distillation_params)
        distillation_params["optimizer"] = stutil.load_optimizer(
            distillation_params["optimizer"],
            self.training_hyperparams["lr"]
        )
        distiller = Distiller(student, teacher)
        distiller.compile(**distillation_params)

        return distiller

    def get_tensorflow_callbacks(self):
        """Prepare the callbacks for the training."""
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_acc", patience=20)
        learning = keras.callbacks.ReduceLROnPlateau(
            monitor="val_acc", factor=0.8, patience=10, min_lr=0.00001
        )

        return [early_stopping, learning]


class OptunaPruner(keras.callbacks.Callback):
    """This is useless since an implementation by the optuna ppl exists already xDD."""
    def __init__(self, trial):
        super(OptunaPruner, self).__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        self.trial.report(logs["val_acc"], epoch)
        if self.trial.should_prune():
            trial = self.trial
            raise optuna.TrialPruned()
