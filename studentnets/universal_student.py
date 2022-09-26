# Implementation of a very simple student network.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL


class UniversalStudent(keras.Model):
    """Optimised DNN for classification on the jet data set used in JEDInet paper.

    This network is implemented based on the publication:
    http://arxiv.org/abs/1908.05318

    Attributes:
        nconst: Number of constituents for the jet data.
        nfeats: Number of features for each constituent.
        activ: Activation function to use between the dense layers.
        name: Name of this network.
    """

    def __init__(
        self, node_size: int = 64, activ: str = "relu", input_dims: tuple = None
        ):

        super(UniversalStudent, self).__init__(name="UniversalStudent")

        self.input_dims = input_dims
        # Hyperparameters decided for universal student.
        self.node_size = node_size
        self.activ = activ
        self.nclasses = 5

        self.__build_network()

    def __build_network(self):
        """Lay out the anatomy of the universal student network."""
        self._dense_layer_1 = KL.Dense(self.node_size)
        if self.input_dims != None:
            self._dense_layer_1 = KL.Dense(self.node_size, input_shape=self.input_dims)
        self._activ_funct_1 = KL.Activation(self.activ)
        self._dense_layer_2 = KL.Dense(int(self.node_size) / 2)
        self._activ_funct_2 = KL.Activation(self.activ)
        self._dense_layer_3 = KL.Dense(int(self.node_size) / 2)
        self._activ_funct_3 = KL.Activation(self.activ)
        self._dense_layer_4 = KL.Dense(self.nclasses)

    def call(self, inputs: np.ndarray, **kwargs):
        inputs = KL.Flatten()(inputs)
        x = self._dense_layer_1(inputs)
        x = self._activ_funct_1(x)
        x = self._dense_layer_2(x)
        x = self._activ_funct_2(x)
        x = self._dense_layer_3(x)
        x = self._activ_funct_3(x)

        logits = self._dense_layer_4(x)

        return logits
