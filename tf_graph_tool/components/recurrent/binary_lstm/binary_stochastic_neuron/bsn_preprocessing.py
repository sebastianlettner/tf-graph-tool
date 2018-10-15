""" Preprocessing describes the process of mapping the input to the neuron to (0, 1). """

import tensorflow as tf
import bsn_literals
from tensorflow.python.framework import ops


class BsnPreprocessing(object):

    """
    The BsnPreprocessing class provides tensor for mapping the neurons input to (0, 1).
    """

    @classmethod
    def produce(cls, x, slope_tensor, tf_graph, method=bsn_literals.PASS_THROUGH_SIGMOID):

        """

        Args:
            x(tensor): Input to the neuron.
            method(str): Method used for the mapping.
            slope_tensor(Tensor): slope adjusts the slope of the sigmoid function
                                  for purposes of the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)
                                  Only necessary for sigmoid method.
        Returns:
            x_processed(tensor): Processed input.
        """

        if method == bsn_literals.PASS_THROUGH_SIGMOID:
            return BsnPreprocessing.pass_through_sigmoid(x, tf_graph)

        elif method == bsn_literals.SIGMOID:
            return BsnPreprocessing.sigmoid(x, slope_tensor=slope_tensor)

        else:
            raise ValueError("Unrecognized preprocessing method.")

    @classmethod
    def pass_through_sigmoid(cls, x, graph):

        """
        Sigmoid that uses the identity function as its gradient.

        Args:
            x(tensor): Input to the neuron

        Returns:
            x_processed(tensor): Input after running through sigmoid with special gradient.

        """

        with ops.name_scope("PassThroughSigmoid") as name:
            with graph.gradient_override_map({"Sigmoid": "Identity"}):
                return tf.sigmoid(x, name=name)

    @classmethod
    def sigmoid(cls, x, slope_tensor):

        """
        Sigmoid function.
        Args:
            x(tensor): Input to the neuron.
            slope_tensor(Tensor): slope adjusts the slope of the sigmoid function
                                  for purposes of the Slope Annealing Trick (see http://arxiv.org/abs/1609.01704)

        Returns:
            x_processed(tensor): Input after running through sigmoid.
        """
        return tf.sigmoid(slope_tensor*x)
