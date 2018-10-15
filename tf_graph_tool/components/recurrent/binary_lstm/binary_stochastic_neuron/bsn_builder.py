" This file implements a builder for BSN Layers. "

import tensorflow as tf
import bsn_literals
from binary_stochastic_neuron import BinaryStochasticST


class BsnBuilder(object):

    @staticmethod
    def bsn_layer(layer_input,
                  name,
                  stochastic_gradient_estimator,
                  slope_tensor,
                  preprocessing_method,
                  stochastic_method,
                  tf_graph,
                  loss_op_name="loss_by_example"
                  ):

        """

        Args:
            num_outputs(int): Number of neurons in the layer
            layer_input(tensor): Input to the neuron
            name(str): Scope of the tensors.
            stochastic_gradient_estimator(str): The estimator for gradient.
            slope_tensor(tensor): Slope factor for slope annealing trick.
                                  Only for straight through estimator.
            preprocessing_method(str): Mapping of the input to (0, 1).
                                        Only for the straight through estimator.
            stochastic_method(str): Method for the sampling.
                                    Only for the straight through estimator.
            loss_op_name(str): Loss of the network must be accessible for the REINFORCE estimator.
                               The loss Tensor must be named with loss_op_name.

        Returns:
            Layer of stochastic binary neurons.
            Same shape as the input.
        """
        with tf.variable_scope(name):

            if stochastic_gradient_estimator == bsn_literals.STRAIGHT_THROUGH:

                return BinaryStochasticST.produce(x=layer_input,
                                                  slope_tensor=slope_tensor,
                                                  preprocessing_method=preprocessing_method,
                                                  stochastic_method=stochastic_method,
                                                  tf_graph=tf_graph)

            else:
                raise Exception("Unknown estimator: " + str(stochastic_gradient_estimator))

