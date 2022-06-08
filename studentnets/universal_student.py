# Implementation of a very simple student network.

from tensorflow import keras
from tensorflow.keras.models import Model
import tensorflow.keras.layers as KL

class UniversalStudent(Model):
    """Simple network that can learn from any teacher through knowledge distillation.
    The principles of knowledge distillation are explained in the following paper:
    http://arxiv.org/abs/1503.02531

    Another application to HEP data can is presented in the publication:
    https://doi.org/10.1140/epjc/s10052-021-09770-w

    Attributes:
        nconst: Number of constituents for the jet data.
        nfeats: Number of features for each constituent.
    """
    def __init__(nconst: int, nfeats: int):
        super(UniversalStudent, self).__init__()

        self._nconst = nconst
        self._nfeats = nfeats

        self.__build_network()


    def __build_network(self):


