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


class DeepSets_Equiv(keras.Model):
    """Deep sets permutation equivariant graph network https://arxiv.org/abs/1703.06114.

    Attributes:
        nnodes_phi: Number of nodes in the layers of the phi neural network.
        nnodes_rho: Number of nodes in the layers of the rho neural network.
        activ: Activation function to use between the dense layers.
    """

    def __init__(self, nnodes_phi: int = 32, nnodes_rho: int = 16, activ: str = "elu"):
        super(DeepSets_Equiv, self).__init__(name="DeepSets_Equiv")
        self.nclasses = 5

        self.phi = keras.Sequential(
            [
                PermutationEquivariantMean(nnodes_phi),
                KL.Activation(activ),
                PermutationEquivariantMean(nnodes_phi),
                KL.Activation(activ),
                PermutationEquivariantMean(nnodes_phi),
                KL.Activation(activ),
            ]
        )

        self.rho = keras.Sequential(
            [KL.Dense(nnodes_rho, activation=activ), KL.Dense(self.nclasses)]
        )

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        sum_output = tf.reduce_mean(phi_output, axis=1)
        rho_output = self.rho(sum_output)

        return rho_output


class DeepSets_Inv(keras.Model):
    """Deep sets permutation invariant graph network https://arxiv.org/abs/1703.06114.

    Attributes:
        nnodes_phi: Number of nodes in the layers of the phi neural network.
        nnodes_rho: Number of nodes in the layers of the rho neural network.
        activ: Activation function to use between the dense layers.
    """

    def __init__(self, nnodes_phi: int = 32, nnodes_rho: int = 16, activ: str = "elu"):
        super(DeepSets_Inv, self).__init__(name="DeepSets_Inv")
        self.nclasses = 5

        self.phi = keras.Sequential(
            [
                KL.Dense(nnodes_phi),
                KL.Activation(activ),
                KL.Dense(nnodes_phi),
                KL.Activation(activ),
                KL.Dense(nnodes_phi),
                KL.Activation(activ),
            ]
        )

        self.rho = keras.Sequential(
            [KL.Dense(nnodes_rho, activation=activ), KL.Dense(self.nclasses)]
        )

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        sum_output = tf.reduce_mean(phi_output, axis=1)
        rho_output = self.rho(sum_output)

        return rho_output
