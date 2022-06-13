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
        self,
        node_size: int = 64,
        activ: str = "relu",
        nclasses: int = 5,
        name: str = "UniversalStudent",
        ):
        super(UniversalStudent, self).__init__(name=name)

        self.activ = activ
        self.nclasses = nclasses
        self.node_size = node_size


        self.__build_network()

    def __build_network(self):
        """Lay out the anatomy of the universal student network."""
        self._batch_norma_1 = KL.BatchNormalization(name="initial_batch_normalisation")
        self._dense_layer_1 = KL.Dense(self.node_size, name="dense_layer_1")
        self._batch_norma_2 = KL.BatchNormalization(name="second_batch_normalisation")
        self._activ_funct_1 = KL.Activation(self.activ, name="activation_1")
        self._dense_layer_2 = KL.Dense(self.node_size/2, name="dense_layer_2")
        self._batch_norma_3 = KL.BatchNormalization(name="third_batch_normalisation")
        self._activ_funct_2 = KL.Activation(self.activ, name="activation_2")
        self._dense_layer_3 = KL.Dense(self.node_size/2, name="dense_layer_3")
        self._batch_norma_4 = KL.BatchNormalization(name="fourth_batch_normalisation")
        self._activ_funct_3 = KL.Activation(self.activ, name="activation_3")
        self._dense_layer_4 = KL.Dense(5, name="dense_layer_4")
        self._activ_funct_4 = KL.Activation(self.activ, name="activation_4")

    def call(self, inputs: np.ndarray, **kwargs):
        """Define the physiology of the universal student. Guide the input through the
        layers of the network and any other miscellaneous operations, such as matrix
        multiplications, tranposes, etc..
        """
        flat_data = KL.Flatten()(inputs)
        norm_data = self._batch_norma_1(flat_data)
        proc_data = self._dense_layer_1(norm_data)
        proc_data = self._batch_norma_2(proc_data)
        proc_data = self._activ_funct_1(proc_data)
        proc_data = self._dense_layer_2(proc_data)
        proc_data = self._batch_norma_3(proc_data)
        proc_data = self._activ_funct_2(proc_data)
        proc_data = self._dense_layer_3(proc_data)
        proc_data = self._batch_norma_4(proc_data)
        proc_data = self._activ_funct_3(proc_data)
        proc_data = self._dense_layer_4(proc_data)
        proc_data = self._activ_funct_4(proc_data)

        return proc_data
