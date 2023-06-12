# Calculates the number of FLOPs in the DeepSets models.

import numpy as np
import functools

import tensorflow as tf
from tensorflow import keras

def dense_flops(layer: ):
    MAC = layer.output_shape[1] * layers.input_shape[1]
    if layers.get_config()["use_bias"]:
        ADD = layer.output_shape[1]
    else:
        ADD = 0
    return MAC*2 + ADD

def get_flops(model: keras.Model):
    """Calculates and returns the number of floating point operations in a DS model."""

    for layer in model.layers:
        if layer.get_config()['name'] == 'sequential':
            for layer in layer['layers']:


def get_flops_sequential(sequential_layer: dict):
    """Calculate number of floating point operations in sequential comb of layers."""
    flops = 0
    input_layer = sequential_layer['layers'][0]
    input_shape = input_layer['config']['batch_input_shape'][1:]
    for layer in sequential_layer['layers'][1:]:
        if layer['class_name'] == 'Dense':
            flops += get_flops_dense(input_shape, layer['config'])
            input_shape[-1] = layer['config']['units']
        elif layer['class_name'] == 'EquivariantMean':
            flops += get_flops_equivariantmean(input_shape, layer['config'])
        elif layer['class_name'] == 'Activation':
            flops += get_flops_activation(input_shape, layer['config'])


def get_flops_dense(input_shape: tuple, dense_layer: dict):
    """Calculates the number of floating point operations in a dense layer."""
    if dense_layer['activation'] != 'linear':
        raise RuntimeError("Activation is not linear. Other types of not supported.")

    units = dense_layer['units']
    MAC = functools.reduce(lambda: x, y: x*y, input_shape)*units

    if dense_layer['use_bias'] == True:
        ADD = units
        return ADD + MAC*2

    return MAC*2


def get_flops_activation(input_shape: tuple, activation: str):
    """Approximates the number of floating point operations in an activation."""
    ninputs = functools.reduce(lambda: x, y: x*y, input_shape)
    activation = activation['activation']

    switcher = {
        "relu": lambda: ninputs,
        "tanh": lambda: ninputs*50,
    }

    activation_flops = switcher.get(activation, lambda: None)()
    if activation_flops == None:
        raise RuntimeError(f"Number of flops calc. not implemented for {activation}.")

    return activation_flops

def get_flops_equivariant():
