# Implementation of the permutation equivariant Deep Sets network from the
# https://arxiv.org/abs/1703.06114 paper.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL


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


class DeepSets(keras.Model):
    """Deep sets permutation equivariant graph network https://arxiv.org/abs/1703.06114.

    The number of parameters is similar to the universal student network.

    Attributes:
        nnodes: Number of constituents for the jet data.
        activ: Activation function to use between the dense layers.
        input_dims: The input dimensions of the network.
    """

    def __init__(self, nnodes: int = 32):
        super(DeepSets, self).__init__(name="DeepSets")
        self.nclasses = 5

        self.phi = keras.Sequential(
            [
                PermutationEquivariantMean(nnodes),
                KL.Activation("elu"),
                PermutationEquivariantMean(nnodes),
                KL.Activation("elu"),
                PermutationEquivariantMean(nnodes),
                KL.Activation("elu"),
            ]
        )

        self.rho = keras.Sequential(
            [KL.Dense(nnodes / 2, activation="elu"), KL.Dense(self.nclasses)]
        )

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        sum_output = tf.reduce_mean(phi_output, axis=1)
        rho_output = self.rho(sum_output)

        return rho_output
