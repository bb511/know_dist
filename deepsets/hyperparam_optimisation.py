# Determined AI DeepSets Trial Class for hyperparameter optimisation.
# See the deepsets.py file for detailed information about the architecture.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
from determined.keras import TFKerasTrial, TFKerasTrialContext

from . import util
from .deepsets import DeepSets
from util.data import Data


class PermutationEquivariantMax(KL.Layer):
    """Permutation equivariant neural network layer with max operation."""

    def __init__(self, dim):
        super(PermutationEquivariantMax, self).__init__()
        self.gamma = KL.Dense(dim)
        self.lambd = KL.Dense(dim, use_bias=False)

    def call(self, inputs: np.ndarray, **kwargs):
        x_maximum = tf.reduce_max(inputs, axis=1, keepdims=True)
        x_maximum = self.lambd(x_maximum)
        x = self.gamma(inputs)
        x = x - x_maximum

        return x


class PermutationEquivariantMean(KL.Layer):
    """Permutation equivariant neural network layer with mean operation."""

    def __init__(self, dim):
        super(PermutationEquivariantMean, self).__init__()
        self.gamma = KL.Dense(dim)
        self.lambd = KL.Dense(dim, use_bias=False)

    def call(self, inputs: np.ndarray, **kwargs):
        x_mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        x_mean = self.lambd(x_mean)
        x = self.gamma(inputs)
        x = x - x_mean

        return x

class DeepSetsTrial(TFKerasTrial):
    def __init__(self, context: TFKerasTrialContext):
        self.context = context
        self.context.configure_fit(shuffle=True)

    def build_model(self):

        nnodes_phi = self.context.get_hparam("nnodes_phi")
        nnodes_rho = self.context.get_hparam("nnodes_rho")
        activation = self.context.get_hparam("activation")

        model = DeepSets(nnodes_phi, nnodes_rho, activation)
        model = self.context.wrap_model(model)

        optimizer = util.choose_optimiser(
            choice=self.context.get_hparam("optimizer"),
            learning_rate=self.context.get_hparam("lr")
            )
        loss = util.choose_loss(self.context.get_hparam("loss"))
        metrics = ["categorical_accuracy"]

        model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        return model

    def build_training_data_loader():
        jet_data = Data.shuffled(
            data_folder="../../ki_data/intnet_input",
            norm="robust",
            train_events=-1,
            test_events=0,
            pt_min="2.0",
            nconstituents="150",
            feature_selection="jedinet",
            flag="",
            )

        return jet_data.tr_data, jet_data.tr_target

    def build_validation_data_loader():
        jet_data = Data.shuffled(
            data_folder="../../ki_data/intnet_input",
            norm="robust",
            train_events=0,
            test_events=-1,
            pt_min="2.0",
            nconstituents="150",
            feature_selection="jedinet",
            flag="",
            )

        return jet_data.te_data, jet_data.te_target


