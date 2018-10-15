""" Wrapper for a conv layer. """

from tf_graph_tool.components.base_config import RootConfig
from tf_graph_tool.components.base_component import BaseComponent
from tf_graph_tool.util.tf_graph_utils import *
from tf_graph_tool.components.component_literals import HIDDEN, OUTPUT


class Conv2DLayerConfig(RootConfig):

    """ Configuration of a Convolution 2D Layer. """

    def __init__(self,
                 builder,
                 name,
                 scope,
                 type,
                 filter,
                 kernel_size,
                 flatten_output=True,
                 strides=(1, 1),
                 padding='same',
                 activations=tf.nn.relu,
                 use_bias=True,
                 kernel_initializer=None,
                 bias_initializer=None,
                 tensorboard_verbosity=0,
                 ):

        """

        Args:
            name(str): Inner scope of the variables .
            scope(str): Outer scope of the variables.
            type(str): Defines the role of the component in the graph. Can be output or hidden.
            filter(int): The dimensionality of the output space
                         (i.e. the number of filters in the convolution).
            kernel_size(): An integer or tuple/list of 2 integers,
                           specifying the height and width of the 2D convolution window.
            flatten_output(bool): Flatten the output of the last layer.
            strides(): An integer or tuple/list of 2 integers,
                       specifying the strides of the convolution along the height and width
            padding(str): 'valid' or 'same'
            activations(func): Activation function. Set it to None to maintain a linear activation.
            use_bias(bool): Boolean, whether the layer uses a bias.
            kernel_initializer(func): An initializer for the convolution kernel.
            bias_initializer(func): An initializer fot the bias
            tensorboard_verbosity(int):
                0: No tensorboard.
                1: Only the activations are tracked.
                2: Activations and weights are tracked. This means the output operation and all the variables.
        """

        self._name = name
        self._scope = scope
        self._tensorboard_verbosity = tensorboard_verbosity
        self._filter = filter
        self._kernel_size = kernel_size
        self._flatten_output = flatten_output
        self._strides = strides
        self._padding = padding
        self._activations = activations
        self._use_bias = use_bias
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer

        super(Conv2DLayerConfig, self).__init__(
            factory_request='conv_2D',
            type=type,
            builder=builder
        )
        super(Conv2DLayerConfig, self).add_node_to_nx_graph(self)

    @property
    def name(self):
        return self._name

    @property
    def scope(self):
        return self._scope

    @property
    def tensorboard_verbosity(self):
        return self._tensorboard_verbosity

    @property
    def filter(self):
        return self._filter

    @property
    def kernel_size(self):
        return self._kernel_size

    @property
    def flatten_output(self):
        return self._flatten_output

    @property
    def stride(self):
        return self._strides

    @property
    def padding(self):
        return self._padding

    @property
    def activations(self):
        return self._activations

    @property
    def use_bias(self):
        return self._use_bias

    @property
    def kernel_initializer(self):
        return self._kernel_initializer

    @property
    def bias_initializer(self):
        return self._bias_initializer


class Conv2DLayer(BaseComponent):

    """ 2D Convolution Layer. """

    def __init__(self, config, inputs):
        """

        Initializes object.

        Args:
            config(Conv2DLayerConfig): Configuration
            inputs(list): List of inout tensors.
        """
        self._inputs = inputs
        self._config = config

        super(Conv2DLayer, self).__init__(
            name=self._config.name,
            output=self.conv2d_layer,
            type=self._config.type,
            scope=self._config.scope,
        )

    @graph_component
    def conv2d_layer(self):

        """

        Returns:

        """

        with tf.variable_scope(self._config.scope):
            with tf.variable_scope(self._config.name):
                conv_layer = tf.layers.conv2d(
                    inputs=tf.concat(self._inputs, 1),
                    filters=self.config.filter,
                    kernel_size=self.config.kernel_size,
                    strides=self.config.stride,
                    padding=self.config.padding,
                    activation=self.config.activations,
                    use_bias=self.config.use_bias,
                    kernel_initializer=self.config.kernel_initializer,
                    bias_initializer=self.config.bias_initializer
                )

                if self.config.flatten_output:
                    conv_layer = tf.reshape(conv_layer, [-1, self.config.filter * self.config.kernel_size[0] *
                                                         self.config.kernel_size[1]])
        # only tensorboard stuff
        if self._config.tensorboard_verbosity == 0:
            return conv_layer

        elif self._config.tensorboard_verbosity == 1:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.histogram(self._config.name + 'activations', conv_layer))
            return conv_layer

        elif self._config.tensorboard_verbosity == 2:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            current_scope = self.config.builder.tf_graph.get_name_scope()
            if current_scope == '':
                scope = self._config.scope + '/' + self._config.name
            else:
                scope = current_scope + '/' + self._config.scope + '/' + self._config.name
            variables = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
            summaries.append(tf.summary.histogram(self._config.name + 'activations', conv_layer))
            summaries += [tf.summary.histogram(var.name, var) for var in variables]
            return conv_layer

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
