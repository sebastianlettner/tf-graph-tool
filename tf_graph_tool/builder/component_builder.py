""" Builds components ."""

import tensorflow as tf
from tf_graph_tool.builder.compute_graph import ComputeGraph
from tf_graph_tool.components.component_factory import ComponentFactory


class ComponentBuilder(object):

    """ This class builds components (e.g. a dense layer) """

    def __init__(self, graph, config_node_id, compute_graph):

        """

        Args:
            graph(nx.DiGraph): The nx graph resembling the tensorflow compute graph.
            compute_graph(ComputeGraph): The neural network this class adds the components to.
        """
        self._graph = graph
        self._compute_graph = compute_graph
        self._config_node_id = config_node_id
        self._config_node = self._graph.node[config_node_id]
        self._configuration = self.config_node['config']
        self._inputs = []

    def build(self):

        """
        This function builds the configured node in tensorflow.
        It uses recursive function calls to collect the inputs for the node.

        Returns:
            output(Tensor): The output tensor of the component

        """

        # if the component is a placeholder, build and return
        if self._configuration.factory_request == 'placeholder':
            component = ComponentFactory.produce(
                config=self._configuration
            )
            self.config_node['output'] = component.output
            self.config_node['is_build'] = True
            self._compute_graph.add_component(component)
            return self.config_node['output']

        # go through the child nodes and build the. this loop implements the recursive mechanism
        for in_edge in self._graph.in_edges(self._config_node_id):
            child_node = self._graph.node[in_edge[0]]
            # the component is already build
            if child_node['is_build']:
                self._inputs.append(child_node['output'])
            else:
                self._inputs.append(child_node['component_builder'].build())

        # we have collected all inputs and use the factory to build this node with tensorflow
        component = ComponentFactory.produce(
            config=self._configuration,
            inputs=self._inputs
        )
        self.config_node['output'] = component.output
        self.config_node['is_build'] = True

        # manage the new component of the tf graph
        self._compute_graph.add_component(component)
        return self.config_node['output']

    @property
    def config_node(self):
        return self._config_node
