# Implementation of a very simple student network.

import numpy as np

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as KL


class JEDIstudent(keras.Model):
    """Optimised DNN for classification on the jet data set used in JEDInet paper.

    This network is implemented based on the publication:
    http://arxiv.org/abs/1908.05318

    Attributes:
        nconst: Number of constituents for the jet data.
        nfeats: Number of features for each constituent.
        activ: Activation function to use between the dense layers.
        name: Name of this network.
    """

    def __init__(self):
        super(JEDIstudent, self).__init__(name="JEDIdnn")

        # Best hyperparameter according to http://arxiv.org/abs/1908.05318.
        self.node_size = 80
        self.activ = "elu"
        self.nclasses = 5
        self.dropout_rate = 0.11

        self.__build_network()

    def __build_network(self):
        """Lay out the anatomy of the universal student network."""
        self._dense_layer_1 = KL.Dense(self.node_size, input_shape=(48, ))
        self._activ_funct_1 = KL.Activation(self.activ)
        self._dropo_layer_1 = KL.Dropout(self.dropout_rate)
        self._dense_layer_2 = KL.Dense(self.node_size)
        self._activ_funct_2 = KL.Activation(self.activ)
        self._dropo_layer_2 = KL.Dropout(self.dropout_rate)
        self._dense_layer_3 = KL.Dense(self.node_size)
        self._activ_funct_3 = KL.Activation(self.activ)
        self._dropo_layer_3 = KL.Dropout(self.dropout_rate)
        self._dense_layer_4 = KL.Dense(self.nclasses)

    def call(self, inputs: np.ndarray, **kwargs):
        inputs = KL.Flatten()(inputs)
        x = self._dense_layer_1(inputs)
        x = self._activ_funct_1(x)
        x = self._dropo_layer_1(x)
        x = self._dense_layer_2(x)
        x = self._activ_funct_2(x)
        x = self._dropo_layer_2(x)
        x = self._dense_layer_3(x)
        x = self._activ_funct_3(x)
        x = self._dropo_layer_3(x)

        logits = self._dense_layer_4(x)

        return logits
