# Barebones implementation of the DeepSets network.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL




def get_deepsets_net():
    deepsets_input = keras.Input(shape=(16, 16), name="jet_event")
    # Permutation Equivariant Layer
    x_max = tf.reduce_max(deepsets_input, axis=1, keepdims=True)
    x_lambd = KL.Dense(32, use_bias=False)(x_max)
    x_gamma = KL.Dense(32)(deepsets_input)
    x = x_gamma - x_lambd
    x = KL.Activation('elu')(x)
    # Permutation Equivariant Layer
    x_max = tf.reduce_max(x, axis=1, keepdims=True)
    x_lambd = KL.Dense(32, use_bias=False)(x_max)
    x_gamma = KL.Dense(32)(x)
    x = x_gamma - x_lambd
    x = KL.Activation('elu')(x)
    # Permutation Equivariant Layer
    x_max = tf.reduce_max(x, axis=1, keepdims=True)
    x_lambd = KL.Dense(32, use_bias=False)(x_max)
    x_gamma = KL.Dense(32)(x)
    x = x_gamma - x_lambd
    x = KL.Activation('elu')(x)

    x = tf.reduce_max(x, axis=1)
    x = KL.Dense(16)(x)
    x = KL.Activation('elu')(x)
    deepsets_output = KL.Dense(5)(x)

    deepsets_network = keras.Model(deepsets_input, deepsets_output, name="deepsets")

    return deepsets_network
