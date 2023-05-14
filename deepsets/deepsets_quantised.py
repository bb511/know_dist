# Quantised deepsets networks equivalent to the float32 implementations in deepsets.py.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL
import qkeras


class EquivariantMaxQuantised(KL.Layer):
    """Permutation equivariant neural network layer with max operation."""

    def __init__(self, dim, nbits):
        super(PermutationEquivariantMax, self).__init__()
        self.gamma = qkeras.QDense(dim, kernel_quantizer=nbits, bias_quantizer=nbits)
        self.lambd = qkeras.QDense(
            dim, use_bias=False, kernel_quantizer=nbits, bias_quantizer=nbits
        )

    def call(self, inputs: np.ndarray, **kwargs):
        x_maximum = tf.reduce_max(inputs, axis=1, keepdims=True)
        x_maximum = self.lambd(x_maximum)
        x = self.gamma(inputs)
        x = x - x_maximum

        return x


class EquivariantMeanQuantised(KL.Layer):
    """Permutation equivariant neural network layer with mean operation."""

    def __init__(self, dim, nbits):
        super(EquivariantMean, self).__init__()
        self.gamma = KL.QDense(dim, kernel_quantizer=nbits, bias_quantizer=nbits)
        self.lambd = KL.QDense(
            dim, use_bias=False, kernel_quantizer=nbits, bias_quantizer=nbits
        )

    def call(self, inputs: np.ndarray, **kwargs):
        x_mean = tf.reduce_mean(inputs, axis=1, keepdims=True)
        x_mean = self.lambd(x_mean)
        x = self.gamma(inputs)
        x = x - x_mean

        return x


class DeepSetsEquivQuantised(keras.Model):
    """Deep sets permutation equivariant graph network https://arxiv.org/abs/1703.06114.

    Attributes:
        nnodes_phi: Number of nodes in the layers of the phi neural network.
        nnodes_rho: Number of nodes in the layers of the rho neural network.
        activ: Activation function to use between the dense layers.
    """

    def __init__(
        self,
        nnodes_phi: int = 32,
        nnodes_rho: int = 16,
        activ: str = "elu",
        nbits: int = 8,
    ):
        super(DeepSetsEquivQuantised, self).__init__(name="DeepSetsEquivQuantised")
        self.nclasses = 5
        nbits = format_quantiser(nbits)
        activ = format_qactivation(activ, nbits)

        self.phi = keras.Sequential(
            [
                EquivariantMeanQuantised(nnodes_phi, nbits),
                qkeras.QActivation(activ),
                EquivariantMeanQuantised(nnodes_phi, nbits),
                qkeras.QActivation(activ),
                EquivariantMeanQuantised(nnodes_phi, nbits),
                qkeras.QActivation(activ),
            ]
        )

        self.rho = keras.Sequential(
            [
                qkeras.QDense(nnodes_rho, kernel_quantizer=nbits, bias_quantizer=nbits),
                qkeras.QActivation(activ),
                qkeras.QDense(
                    self.nclasses, kernel_quantizer=nbits, bias_quantizer=nbits
                ),
            ]
        )

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        sum_output = tf.reduce_mean(phi_output, axis=1)
        rho_output = self.rho(sum_output)

        return rho_output


class DeepSetsInvQuantised(keras.Model):
    """Deep sets permutation invariant graph network https://arxiv.org/abs/1703.06114.

    Attributes:
        nnodes_phi: Number of nodes in the layers of the phi neural network.
        nnodes_rho: Number of nodes in the layers of the rho neural network.
        activ: Activation function to use between the dense layers.
    """

    def __init__(
        self,
        nnodes_phi: int = 32,
        nnodes_rho: int = 16,
        activ: str = "elu",
        nbits: int = 8,
    ):
        super(DeepSetsInvQuantised, self).__init__(name="DeepSetsInvQuantised")
        self.nclasses = 5
        nbits = format_quantiser(nbits)
        activ = format_qactivation(activ, nbits)

        self.phi = keras.Sequential(
            [
                qkeras.QDense(nnodes_phi, kernel_quantizer=nbits, bias_quantizer=nbits),
                qkeras.QActivation(activ),
                qkeras.QDense(nnodes_phi, kernel_quantizer=nbits, bias_quantizer=nbits),
                qkeras.QActivation(activ),
                qkeras.QDense(nnodes_phi, kernel_quantizer=nbits, bias_quantizer=nbits),
                qkeras.QActivation(activ),
            ]
        )

        self.rho = keras.Sequential(
            [
                qkeras.QDense(nnodes_rho, kernel_quantizer=nbits, bias_quantizer=nbits),
                qkeras.QActivation(activ),
                qkeras.QDense(
                    self.nclasses, kernel_quantizer=nbits, bias_quantizer=nbits
                ),
            ]
        )

    def call(self, inputs: np.ndarray, **kwargs):
        phi_output = self.phi(inputs)
        sum_output = tf.reduce_mean(phi_output, axis=1)
        rho_output = self.rho(sum_output)

        return rho_output


def format_quantiser(self, nbits: int):
    """Format the quantisation of the ml floats in a QKeras way."""
    if nbits == 1:
        self.nbits = "binary(alpha=1)"
    elif nbits == 2:
        self.nbits = "ternary(alpha=1)"
    else:
        self.nbits = f"quantized_bits({nbits}, 0, alpha=1)"


def format_qactivation(self, activation: str, nbits: int) -> str:
    """Format the activation function strings in a QKeras friendly way."""
    return f"quantized_{activation}({nbits}, 0)"
