""" Fully connected layer. """

from tf_graph_tool.util.tf_graph_utils import *
from tf_graph_tool.components.base_config import RootConfig
from tf_graph_tool.components.base_component import BaseComponent
from tf_graph_tool.components.component_literals import HIDDEN, OUTPUT


class DenseLayerConfig(RootConfig):

    """ Dense layer configuration . """

    def __init__(self,
                 builder,
                 name,
                 scope,
                 num_units,
                 type=HIDDEN,
                 activation=None,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=tf.zeros_initializer(),
                 tensorboard_verbosity=0,
                 ):

        """

        Initializes object.

        Args:
            name(str): Name of the component. The inner scope
            scope(str): Scope of the component. The outer scope
            num_units(int): Number of hidden units.
            type(str): Defines the role of the component in the graph. Can be output or hidden.
            activation(func): The activation function.
            use_bias(bool): if true, a bias term is used
            kernel_initializer(func): Initializer function for the weights.
            bias_initializer(func): Initializer function for the bias
            tensorboard_verbosity(int):
                0: No tensorboard.
                1: Only the activations are tracked.
                2: Activations and weights are tracked. This means the output operation and all the variables.

        """
        self._name = name
        self._scope = scope
        self._num_units = num_units
        self._activation = activation
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._tensorboard_verbosity = tensorboard_verbosity

        super(DenseLayerConfig, self).__init__(
            type=type,
            factory_request='dense',
            builder=builder
        )
        super(DenseLayerConfig, self).add_node_to_nx_graph(self)

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def num_units(self):
        return self._num_units

    @property
    def activation(self):
        return self._activation

    @property
    def use_bias(self):
        return self._use_bias

    @property
    def kernel_initializer(self):
        return self._kernel_initializer

    @property
    def bias_initializer(self):
        return self._bias_initializer

    @property
    def tensorboard_verbosity(self):
        return self._tensorboard_verbosity


class DenseLayer(BaseComponent):

    """ Densely-connected layer. """

    def __init__(self,
                 config,
                 inputs
                 ):
        """

               Initializes object.

               Args:
                   config(DenseLayerConfig): Configuration
                   inputs(tensor): The input to the dense layer.

               """
        self._inputs = inputs
        self._config = config
        super(DenseLayer, self).__init__(name=self._config.name,
                                         output=self.dense_layer,
                                         type=self._config.type,
                                         scope=self._config.scope,
                                         )

    @graph_component
    def dense_layer(self):

        """

        Create a dense layer.
        This function also handles the tensorboard.

        Returns:

        """

        with tf.variable_scope(self._config.scope):
            with tf.variable_scope(self._config.name):
                layer = tf.layers.dense(
                    inputs=tf.concat(self._inputs, axis=1),
                    units=self._config.num_units,
                    activation=self._config.activation,
                    use_bias=self._config.use_bias,
                    kernel_initializer=self._config.kernel_initializer,
                    bias_initializer=self._config.bias_initializer
                )

        # only tensorboard stuff
        if self._config.tensorboard_verbosity == 0:
            return layer

        elif self._config.tensorboard_verbosity == 1:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.histogram(self._config.name + '_activations', layer))
            return layer

        elif self._config.tensorboard_verbosity == 2:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            current_scope = self.config.builder.tf_graph.get_name_scope()
            if current_scope == '':
                scope = self._config.scope + '/' + self._config.name
            else:
                scope = current_scope + '/' + self._config.scope + '/' + self._config.name
            variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            summaries.append(tf.summary.histogram(self._config.name + '_activations', layer))
            summaries += [tf.summary.histogram(var.name, var) for var in variables]
            return layer

        else:
            raise ValueError("Unsupported tensorboard verbosity {}. Supported are 0, 1, 2.".format(
                self._config.tensorboard_verbosity))

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

        if self.config.type == HIDDEN:
            compute_graph.hidden[self.name] = self.output
        elif self.config.type == OUTPUT:
            compute_graph.outputs[self.name] = self.output
        else:
            raise ValueError('Unknown component type: {}'.format(self.config.type))
