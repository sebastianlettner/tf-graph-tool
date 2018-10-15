""" Neural Network """

import tensorflow as tf
from tf_graph_tool.util.tf_graph_utils import *


class ComputeGraph(object):

    """ This class holds the tensorflow graph. It also provides some interfaces and functions to use the graph """

    def __init__(self, main_scope):

        """
        Initializes object.

        Args:
            main_scope(str): The most outer variable scope for the graph used by this Neural Network class
        """
        self._tensorflow_graph = tf.Graph()
        self._inputs = {}  # for placeholders
        self._hidden = {}  # for everything in between
        self._outputs = {}  # for all output nodes
        self._losses = {}  # for loss operations
        self._main_scope = main_scope
        self.session = None
        self.recurrent_cells = []

    def add_component(self, component):
        """
        This function is called during the building process. Components can place operations in the dictionaries.
        Args:
            component:

        Returns:

        """

        component.decorate_graph(self)

    def create_session(self, **kwargs):
        """
        Create the tensorflow session
        """
        self.session = tf.Session(graph=self.tf_graph, **kwargs)

    @use_network_graph
    def initialize_graph_variables(self):

        """

        This function initializes all variables in the graph.

        Returns:

        """
        self.session.run(tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        ))

    @property
    def inputs(self):
        return self._inputs

    @property
    def hidden(self):
        return self._hidden

    @property
    def outputs(self):
        return self._outputs

    @property
    def main_scope(self):
        return self._main_scope

    @property
    def tf_graph(self):
        return self._tensorflow_graph

    @property
    def losses(self):
        return self._losses

    def get_component(self, name):

        """

        Searches the dictionary for a component with the name.

        Args:
            name(str): The components name

        Returns:
            component
        """

        if name in self.inputs:
            return self.inputs[name]
        elif name in self.hidden:
            return self.hidden[name]
        elif name in self.outputs:
            return self.outputs[name]
        elif name in self.losses:
            return self.losses[name]
        else:
            raise KeyError('There is no component with name {} in the graph'.format(name))

    def get_feed_dict(self, inputs):
        """
        The functions accepts a dictionary where the keys are the names (str) of placeholders in the graph
        and the values of the dictionary are the values you want to feed into the graph.
        The functions replaces the keys (str) with the actual placeholders from the tf graph.
        The output can be used as the feed_dict for the session.run() function from tensorflow.
        Args:
            inputs(dict): A placeholder name to value mapping. This function will then replace the
                          placeholder name with the actual placeholder.

        Returns:

        """
        return {
            self.inputs[key]: inputs[key] for key in inputs.keys()
            if key in self.inputs.keys()
        }

    def run_single_operation(self, op_name, inputs):
        """
        Run a single operation of the graph.

        Args:
            op_name(str): The name the operation has in the tf graph builder configuration.
                          This is the nodes name you defined by implementing the compute graph.
            inputs(dict): Dictionary were the keys are the names (str) of the placeholder nodes in the tf graph builder.
                          The values are the numerical inputs ofr tf graph.
                          of the compute graph. Of course, all the necessary placeholder for running the desired op
                          must be provided.

        Returns:
            output(dict): of the operation.
                          The key is the name of the operation that has ben run. The value is the actual output of the
                          the operation.
        """

        # get the tensorflow operation
        operation = self.get_component(name=op_name)

        # prepare the feed dict
        feed_dict = self.get_feed_dict(inputs=inputs)

        # run the op
        output = self.session.run(
            operation,
            feed_dict=feed_dict
        )
        return output

    def run_multiple_operations(self, op_names, inputs):

        """
        Run multiple operations at once.

        Args:
            op_names(list): Names of the operations in the compute graph wrapper.
            inputs(dict): Dictionary were the keys are the names (str) of the placeholder nodes in the tf graph builder.
                          The values are the numerical inputs ofr tf graph.
                          of the compute graph. Of course, all the necessary placeholder for running the desired op
                          must be provided.

        Returns:
            output(dict): of the operation.
              The keys are the names of the operations that has ben run. The values are the actual outputs of these
              operations.

        """
        # get dict of tensorflow operations
        operations = {op_name: self.get_component(op_name) for op_name in op_names}

        # prepare feed dict
        feed_dict = self.get_feed_dict(inputs=inputs)

        outputs = self.session.run(
            operations,
            feed_dict=feed_dict
        )
        return outputs

