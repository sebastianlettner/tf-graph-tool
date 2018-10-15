""" Configuration for graph components. """

import networkx as nx
from tf_graph_tool.builder.component_builder import ComponentBuilder
from tf_graph_tool.builder.compute_graph import ComputeGraph


class RootConfig(object):

    """ Base Class for configurations of the tensorflow graph components. """

    def __init__(self,
                 builder,
                 factory_request,
                 type,
                 ):

        """

        Initializes object

        Args:
            builder(GraphBuilder): The builder
            factory_request(str): Request for the Component Factory.
            type(str): Type of the components. This can be input, hidden or output.

        """
        self._builder = builder
        self._factory_request = factory_request
        self._type = type

    @property
    def type(self):
        return self._type

    @property
    def builder(self):
        return self._builder

    @property
    def factory_request(self):
        return self._factory_request

    def add_node_to_nx_graph(self, config):
        """
        Adds configuration node to the networkx graph.
        Args:
            config:

        Returns:

        """
        node_id = config.name
        self._builder.nx_graph.add_node(node_id)
        self._builder.nx_graph.node[node_id]['config'] = config
        self._builder.nx_graph.node[node_id]['is_build'] = False
        self._builder.nx_graph.node[node_id]['output'] = None
        self._builder.nx_graph.node[node_id]['component_builder'] = ComponentBuilder(
            graph=self._builder.nx_graph,
            config_node_id=node_id,
            compute_graph=self._builder.compute_graph
        )
        return self._builder.nx_graph.node[node_id]
