# Implementation of a very simple student network.

import numpy as np

from tensorflow import keras
import tensorflow.keras.layers as KL


class UniversalStudent(keras.Model):
    """Simple network that can learn from any teacher through knowledge distillation.
    The principles of knowledge distillation are explained in the following paper:
    http://arxiv.org/abs/1503.02531

    Another application to HEP data can is presented in the publication:
    https://doi.org/10.1140/epjc/s10052-021-09770-w

    Attributes:
        nconst: Number of constituents for the jet data.
        nfeats: Number of features for each constituent.
        activ: Activation function to use between the dense layers.
        name: Name of this network.
    """
    def __init__(
        nconst: int,
        nfeats: int,
        activ: str = "relu",
        name: str = "Universal Student",
        ):
        super(UniversalStudent, self).__init__(name=name)

        self.activ = activ

        self._nconst = nconst
        self._nfeats = nfeats

        self.__build_network()

    def __build_network(self):
        """Lay out the anatomy of the universal student network.
        """
        self._batch_norma_1 = KL.BatchNormalization(name="Initial Batch Normalisation")
        self._dense_layer_1 = KL.Dense(self.node_size, name="Dense Layer 1")
        self._batch_norma_2 = KL.BatchNormalization(name="Second Batch Normalisation")
        self._activ_funct_1 = KL.Activation(self.activ, name="Activation 1")
        self._dense_layer_2 = KL.Dense(self.node_size, name="Dense Layer 2")

    def call(self, inputs: np.ndarray):
        """Define the physiology of the universal student. Guide the input through the
        layers of the network and any other miscellaneous operations, such as matrix
        multiplications, tranposes, etc..
        """
        flat_data = KL.Flatten()(inputs)
        norm_data = self._batch_norm(flat_data)
