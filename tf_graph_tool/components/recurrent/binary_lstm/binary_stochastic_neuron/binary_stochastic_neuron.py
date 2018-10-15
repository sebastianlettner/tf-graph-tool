""" This file implements the Binary Stochastic Neuron. """

import tensorflow as tf
import bsn_literals
from bsn_preprocessing import BsnPreprocessing
from bsn_stochastic_model import BsnStochasticModel
from tensorflow.python.framework import ops


__all__ = ['BinaryStochasticST']


class BinaryStochasticNeuron(object):

    pass


class BinaryStochasticST(BinaryStochasticNeuron):

    @classmethod
    def produce(cls,
                x,
                slope_tensor,
                preprocessing_method,
                stochastic_method,
                tf_graph):

        """

        Args:
            x(tensor): Input to the neuron.
            slope_tensor(tensor): Slope factor for slope annealing trick.
                                  Only necessary if preprocessing method is sigmoid function.
            preprocessing_method(str): Method for preprocessing i.e. mapping to (0, 1)
            stochastic_method(str): Method for sampling.

        Returns:
            bsn(tensor): Binary Stochastic Neuron with specified tensors.
        """

        x_processed = BsnPreprocessing.produce(x,
                                               method=preprocessing_method,
                                               slope_tensor=slope_tensor,
                                               tf_graph=tf_graph)
        return BsnStochasticModel.produce(x_processed,
                                          method=stochastic_method,
                                          tf_graph=tf_graph)

