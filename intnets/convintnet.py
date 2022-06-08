# Interaction net implementation in tensorflow using quantised convolutional layers.

import numpy as np
import itertools
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers as KL


class ConvIntNet(Model):
    """Interaction network implemented with convolutional layers.

    See the following github repositories for more details:
    https://bit.ly/3PhpTcB
    https://bit.ly/39qPL55
    https://bit.ly/3FNQRUI

    For theoretical explanations, please see the papers:
    https://arxiv.org/abs/1612.00222
    https://arxiv.org/abs/1908.05318

    Attributes:
        nconst: Number of constituents the jet data has.
        nclasses: Number of classes the data has.
        nfeats: Number of features the data has.
        *_structure: List containing the number of nodes for each hidden layer.
            If they are empty, they default to on hidden layer with nfeats/2 nodes.
        neffects: Number of effects for your system, i.e., the dimension of the
            output of the fr network.
        ndynamics: Number of dynamical variables in your system, i.e., the output
            dimension for the fo network.
        *_activation: The activation function after each layer for each network.
        summation: Whether a summation layer should be used before the final classifier
            of the network (True) or the 'dynamics' matrix is just flattened (False).
    """

    def __init__(
        self,
        nconst: int,
        nfeats: int,
        nclasses: int = 5,
        fr_structure: list = [30, 15],
        fo_structure: list = [45, 22],
        fc_structure: list = [48],
        neffects: int = 6,
        ndynamics: int = 6,
        fr_activation: str = "relu",
        fo_activation: str = "relu",
        fc_activation: str = "relu",
        summation: bool = True,
    ):

        super().__init__()

        self.nconst = nconst
        self.nedges = nconst * (nconst - 1)
        self.nfeats = nfeats
        self.nclass = nclasses

        self._fr_structure = fr_structure
        self._fo_structure = fo_structure
        self._fc_structure = fc_structure
        self._summation = summation

        self._neffects = neffects
        self._ndynamic = ndynamics

        self._fr_activation = fr_activation
        self._fo_activation = fo_activation
        self._fc_activation = fc_activation

        self._receiver_matrix, self._sender_matrix = self._build_relation_matrices()
        self._build_intnet()

    def _build_intnet(self):
        # First, pass data through batch normalisation layer.
        self._batch_norm = KL.BatchNormalization()
        # Build the MLPs.
        self._fr = self._build_fr()
        self._fo = self._build_fo()
        self._fc = self._build_fc()

    def _build_relation_matrices(self):
        """Construct the relation matrices between the graph nodes."""
        receiver_matrix = np.zeros([self.nconst, self.nedges], dtype=np.float32)
        sender_matrix = np.zeros([self.nconst, self.nedges], dtype=np.float32)
        receiver_sender_list = [
            node
            for node in itertools.product(range(self.nconst), range(self.nconst))
            if node[0] != node[1]
        ]

        for idx, (receiver, sender) in enumerate(receiver_sender_list):
            receiver_matrix[receiver, idx] = 1
            sender_matrix[sender, idx] = 1

        return receiver_matrix, sender_matrix

    def _build_fr(self) -> keras.Sequential:
        """Construct the MLP that computes the effects from the relations of nodes."""
        self._fr_structure.append(self._neffects)

        fr = keras.Sequential(name="fr")
        for idx, layer_nodes in enumerate(self._fr_structure):
            fr.add(KL.Conv1D(layer_nodes, kernel_size=1))
            fr.add(KL.Activation(self._fr_activation))

        return fr

    def _build_fo(self) -> keras.Sequential:
        """Construct the MLP that computes the dynamics from the effects of nodes."""
        self._fo_structure.append(self._ndynamic)

        fo = keras.Sequential(name="fo")
        for idx, layer_nodes in enumerate(self._fo_structure):
            fo.add(KL.Conv1D(layer_nodes, kernel_size=1))
            fo.add(KL.Activation(self._fo_activation))

        return fo

    def _build_fc(self) -> keras.Sequential:
        """Construct the MLP that maps the dynamics to a certain target class."""
        self._fc_structure.append(self.nclass)

        fc = keras.Sequential(name="fc")
        for idx, layer_nodes in enumerate(self._fc_structure):
            fc.add(KL.Dense(layer_nodes))
            if idx == len(self._fc_structure) - 1:
                break
            fc.add(KL.Activation(self._fc_activation))

        fc.add(KL.Activation("softmax"))

        return fc

    def _tmul(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Makes matrix multiplication of shape (I * J * K)(K * L) -> I * J * L."""
        x_shape = tf.shape(x)
        y_shape = tf.shape(y)
        return tf.reshape(
            tf.matmul(tf.reshape(x, [-1, x_shape[2]]), y), [-1, x_shape[1], y_shape[1]]
        )

    def call(self, inputs, **kwargs):
        """Construct the network and direct the flow of the input."""
        # Determine input shape and normalise the input data.
        batch_norm = self._batch_norm(inputs)
        batch_norm = KL.Permute((2, 1), input_shape=batch_norm.shape[1:])(batch_norm)

        # Construct the receiver and sender states matrices by multiplying the input
        # states with the receiver and sender matrices. Need to use convolutional layers
        # to do matrix multiplication since QKeras does not have quantized matrix mul.
        receiver_matrix = self._tmul(batch_norm, self._receiver_matrix)
        sender_matrix = self._tmul(batch_norm, self._sender_matrix)

        # Construct the receiver-sender matrix from which effects are computed.
        rs_matrix = KL.Concatenate(axis=1)([receiver_matrix, sender_matrix])
        del receiver_matrix, sender_matrix

        # Pass receiver-sender matrix through the fr neural net.
        fr_input = KL.Permute((2, 1), input_shape=rs_matrix.shape[1:])(rs_matrix)
        fr_output = self._fr(fr_input)

        # Multiply the effects matrix with the receiver binary matrix.
        fr_output = KL.Permute((2, 1))(fr_output)
        transpose_receiver_matrix = tf.transpose(self._receiver_matrix)
        effects_matrix = self._tmul(fr_output, transpose_receiver_matrix)
        del fr_output

        # Pass the effects matrix through the fo neural net.
        fo_input = KL.Concatenate(axis=1)([batch_norm, effects_matrix])
        del effects_matrix
        fo_input = KL.Permute((2, 1), input_shape=fo_input.shape[1:])(fo_input)
        fo_output = self._fo(fo_input)

        # Pass the dynamics matrix through the fc neural net.
        if self._summation == True:
            fc_input = tf.reduce_sum(fo_output, 1)
        else:
            fc_input = KL.Flatten()(fo_output)

        del fo_output
        fc_output = self._fc(fc_input)

        return fc_output
