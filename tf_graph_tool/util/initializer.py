""" Implementing different initialzers for weights. """

import numpy as np
import tensorflow as tf


def normalized_columns_initializer(std=1.0, seed=1):
    random = np.random.RandomState(seed=seed)

    def _initializer(shape, dtype=None, partition_info=None):
        out = random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer
