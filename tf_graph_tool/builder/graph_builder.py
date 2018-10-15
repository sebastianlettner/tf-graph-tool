""" The Builder for Neural Networks. """

import networkx as nx
import tensorflow as tf
from tf_graph_tool.builder.compute_graph import ComputeGraph



class GraphBuilder(object):

    """ The Builder """

    def __init__(self, main_scope):

        """
        Initializes object.
        Args:
            main_scope(str): The most outer variable scope
        """

        self._nx_graph = nx.DiGraph()
        self._compute_graph = ComputeGraph(main_scope=main_scope)
        self.tf_graph = self._compute_graph.tf_graph

    def build_graph(self, start_nodes):
        """
        Start the recursive mechanism
        Args:
            start_nodes(list): List of end nodes. At these nodes the recursion is started.

        Returns:
            The tensorflow graph wrapped in a compute graph
        """
        with self.compute_graph.tf_graph.as_default():
            with tf.variable_scope(self.compute_graph.main_scope):
                for start_node in start_nodes:
                    start_node['component_builder'].build()

        return self.compute_graph

    def define_graph(self):
        """
        This method implements the concrete graph structure using networkx.
        Returns:
            start_nodes(list)
        """
        raise NotImplementedError

    @property
    def nx_graph(self):
        return self._nx_graph

    @property
    def compute_graph(self):
        return self._compute_graph

    def add_edge(self, node1, node2):
        """

        Args:
            node1(): Identifier of node one. The name of the component
            node2: Identifier of node two. The name of the component

        Returns:

        """
        _ = self.nx_graph.node[node1]
        _ = self.nx_graph.node[node2]

        self.nx_graph.add_edge(node1, node2)
