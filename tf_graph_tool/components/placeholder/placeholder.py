""" Wrapper for tensorflow placeholder """

from tf_graph_tool.components.component_literals import INPUT
from tf_graph_tool.util.tf_graph_utils import *
from tf_graph_tool.components.base_component import BaseComponent
from tf_graph_tool.components.base_config import RootConfig


class PlaceholderConfig(RootConfig):

    def __init__(self,
                 builder,
                 shape,
                 name,
                 tensorboard_verbosity=0,
                 dtype=tf.float32,
                 ):

        """

        Initializes object.

        Args:
            shape(list): Shape of the placeholder.
            name(str): Name of the placeholder.
            dtype(str): The data type

        """
        self._shape = shape
        self._dtype = dtype
        self._name = name
        self._tensorboard_verbosity = tensorboard_verbosity

        super(PlaceholderConfig, self).__init__(type=INPUT,
                                                factory_request='placeholder',
                                                builder=builder)
        super(PlaceholderConfig, self).add_node_to_nx_graph(self)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self):
        return self._name

    @property
    def tensorboard_verbosity(self):
        return self._tensorboard_verbosity


class Placeholder(BaseComponent):

    """ Thin wrapper for a tensorflow placeholder. """

    def __init__(self, config):

        """

        Initializes object.

        Args:
            config(PlaceholderConfig): Configuration.
        """

        self._config = config

        super(Placeholder, self).__init__(name=self.config.name, output=self.placeholder, scope=None,
                                          type=INPUT)

    @graph_component
    def placeholder(self):

        placeholder = tf.placeholder(dtype=self.config.dtype, shape=self.config.shape, name=self.config.name)

        if self.config.tensorboard_verbosity:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            if self.config.shape==[]:
                summaries.append(tf.summary.scalar(self._config.name, placeholder))
            else:
                summaries.append(tf.summary.histogram(self._config.name, placeholder))

        return placeholder

    @property
    def config(self):
        return self._config

    def decorate_graph(self, compute_graph):

        """
        Add the output operation to the neural networks dictionary to make access easier.

        Args:
            compute_graph:

        Returns:

        """

        compute_graph.inputs[self.name] = self.output
