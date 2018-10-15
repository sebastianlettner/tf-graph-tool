""" The stochastic model of the neuron. """

import tensorflow as tf
from tensorflow.python.framework import ops
import bsn_literals


class BsnStochasticModel(object):

    """
    The BsnStochasticModel class provides tensors for sampling the preprocessed input ranging
    from (0, 1) to either 0 or 1.
    THere is also a round operation provided which is of course not stochastic.
    """

    @classmethod
    def produce(cls, x, tf_graph, method=bsn_literals.BERNOULLI):

        """

        Args:
            x(tensor): Preprocessed input of the neuron.
            method(str): Method used for the mapping.

        Returns:
            x_discrete(tensor): Tensor which will evaluate to either 0 or 1.

        """

        if method == bsn_literals.BERNOULLI:
            return BsnStochasticModel.bernoulli(x, tf_graph)

        elif method == bsn_literals.ROUND:
            return BsnStochasticModel.round(x, tf_graph)

        else:
            raise ValueError("Unrecognized method for stochastic sampling: " + method)

    @classmethod
    def bernoulli(cls, x, graph):

        """
        Uses a tensor whose values are in (0,1) to sample a tensor with values in {0, 1}.

        E.g.:
        if x is 0.6, bernoulliSample(x) will be 1 with probability 0.6, and 0 otherwise,
        and the gradient will be pass-through (identity).

        Args:
            x(tensor): The tensor we want to round

        Returns:
            x_sampled(tensor): Mapped tensor.
        """
        with ops.name_scope("BernoulliSample") as name:
            with graph.gradient_override_map({"Ceil": "Identity", "Sub": "BernoulliSample_ST"}):
                return tf.ceil(x - tf.random_uniform(tf.shape(x)), name=name)

    @classmethod
    def round(cls, x, graph):

        """
        Rounds a tensor whose values are in (0,1) to a tensor with values in {0, 1},
        using the straight through estimator for the gradient.

        Args:
            x(tensor): The tensor we want to round

        Returns:
            x_rounded(tensor): Rounded tensor.

        NOTE: Not a stochastic operation. Just for the purpose of experimenting.
        """
        with ops.name_scope("BinaryRound") as name:
            with graph.gradient_override_map({"Round": "Identity"}):
                return tf.round(x, name=name)


@ops.RegisterGradient("BernoulliSample_ST")
def bernoulli_sample_st(op, grad):
    return [grad, tf.zeros(tf.shape(op.inputs[1]))]
