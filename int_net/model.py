# Interaction net implementation in tensorflow using quantised convolutional layers.

import numpy as np
import itertools

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import utils
import qkeras

class QConvIntNet(models.Model):
    """Interaction network implemented with convolutional layers.

    See the following githubrepositories for more details:
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
        nbits: The number of bits to quantise the floats to.
    """
    def __init__(
        self,
        nconst: int = 8,
        nclasses: int = 5,
        nfeats: int = 3,
        fr_structure: list = [],
        fo_structure: list = [],
        fc_structure: list = [],
        neffects: int = 6,
        ndynamics: int = 6,
        fr_activation: str = 'selu',
        fo_activation: str = 'selu',
        fc_activation: str = 'selu',
        nbits: int = 8):


        self.nconst = nconst
        self.nedges = nconst*(nconst - 1)
        self.nfeats = nfeats
        self.nclass = nclasses

        self._fr_structure = fr_structure
        self._fo_structure = fo_structure
        self._fc_structure = fc_structure

        self._neffects = neffects
        self._ndynamic = ndynamics

        self._fr_activation = self._initialise_qactivation(fr_activation, nbits)
        self._fo_activation = self._initialise_qactivation(fo_activation, nbits)
        self._fc_activation = self._initialise_qactivation(fc_activation, nbits)

        self.qbits = self._initialise_quantiser(nbits)

        self._receiver_matrix, self._sender_matrix = self._build_relation_matrices()

        self._fr = self._build_fr()
        self._fo = self._build_fo()
        self._fc = self._build_fc()

    def _initialise_quantiser(qbits: int):
        """Format the quantisation of the ml floats in a QKeras way."""
        if nbits == 1:
            self.qbits = 'binary(alpha=1)'
        elif nbits == 2:
            self.qbits = 'ternary(alpha=1)'
        else:
            self.qbits = f'quantized_bits({nbits}, 0, alpha=1)'

    def _initialise_qactivation(activation: str, nbits: int) -> str:
        """Format the activation function strings in a QKeras friendly way."""
        return f"quantised_{activation}({nbits}, 0)"

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

        fr = keras.Sequential()
        for idx, layer_nodes in enumerate(self._fr_structure):
            fr.add(qkeras.QConv1D(layer_nodes[idx], kernel_size=1,
                                  kernel_quantizer=self.qbits,
                                  bias_quantizer=self.qbits,
                                  name=f'conv1D_fr_{idx}'))
            fr.add(qkeras.QActivation(self._fr_activation, name=f'fr_activ_{idx}'))

        return fr

    def _build_fo(self) -> keras.Sequential:
        """Construct the MLP that computes the dynamics from the effects of nodes."""
        self._fo_structure.append(self._ndynamic)

        fo = keras.Sequential()
        for idx, layer_nodes in enumerate(self._fo_structure):
            fo.add(qkeras.QConv1D(layer_nodes[idx], kernel_size=1,
                                  kernel_quantizer=self.qbits,
                                  bias_quantizer=self.qbits,
                                  name=f'conv1D_fo_{idx}'))
            fo.add(qkeras.QActivation(self._fo_activation, name=f'fo_activ_{idx}'))

        return fo

    def _build_fc(self) -> keras.Sequential:
        """Construct the MLP that maps the dynamics to a certain target class."""
        self._fc_structure.append(self._nclass)

        fc = keras.Sequential()
        for idx, layer_nodes in enumerate(self._fc_structure):
            fc.add(qkeras.QDense(layer_nodes[idx], kernel_quantizer=self.qbits,
                                 bias_quantizer=self.qbits, name=f'fc_dense_{idx}'))
            if idx == len(self._fc_structure - 1): break
            fc.add(qkeras.QActivation(self._fc_activation, name=f'fc_activ_{idx}'))

        fc.add(qkeras.QActivation('softmax', name='softmax_fc'))

        return fc

    def call(self, **kwargs):
        """Construct the network and direct the flow of the input."""
        # Determine input shape and normalise the input data.
        net_input = keras.Input(shape=(self._nconst, self._nfeats), name='net_input')
        bnorm = keras.BatchNormalization(name='normed_input')(net_input)
        bnorm = keras.Permute((2, 1), input_shape=self._bnorm.shape[1:])(bnorm)

        # Construct the receiver and sender states matrices by multiplying the input
        # states with the receiver and sender matrices. Need to use convolutional layers
        # to do matrix multiplication since QKeras does not have quantized matrix mul.
        receivers = qkeras.QConv1D(
            self._receiver_matrix[1], kernel_size=1, kernel_quantizer=self.qbits,
            bias_quantizer=self.qbits, use_bias=False, trainable=False, name='ORr'
        )(bnorm)
        senders = qkeras.QConv1D(
            self._sender_matrix[1], kernel_size=1, kernel_quantizer=self.qbits,
            bias_quantizer=self.qbits, use_bias=False, trainable=False, name='ORs'
        )(bnorm)

        # Construct the receiver-sender matrix from which effects are computed.
        rs_matrix = keras.Concatenate(axis=1)([receivers, senders])
        del receivers, senders

        # Pass receiver-sender matrix through the fr neural net.
        fr_input = keras.Permute((2, 1), input_shape=rs_matrix[1:])(rs)
        fr_output = self._fr(fr_input)
        fr_output = keras.Permute((2, 1))(fr_output)

        # Multiply the effects matrix with the receiver binary matrix.
        effects_matrix = qkeras.QConv1D(
            np.transpose(self._receiver_matrix).shape[1],
            kernel_size=1, kernel_quantizer=self.qbits, bias_quantizer=self.qbits,
            use_bias=False, trainable=False, name='Ebar')
        del fr_output

        # Pass the effects matrix through the fo neural net.
        fo_input = keras.Concatenate(axis=1)([bnorm, effects_matrix])
        del effects_matrix
        fo_input = keras.Permute((2, 1), input_shape=fo_input.shape[1:])(fo_input)
        fo_output = self._fo(fo_input)

        # Pass the dynamics matrix through the fc neural net.
        fc_input = keras.Flatten()(fo_output)
        del fo_output
        fc_output = self._fc(fc_input)

        return fc_output
