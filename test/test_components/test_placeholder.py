""" Testing the plasceholder. """


import unittest
import tensorflow as tf
from tf_graph_tool.components.placeholder.placeholder import PlaceholderConfig
from tf_graph_tool.components.base_config import RootConfig
from tf_graph_tool.builder.graph_builder import GraphBuilder


class BuildPlaceholder(GraphBuilder):

    """ Class for testing """

    def __init__(self, main_scope='test_pl'):

        """

        Args:
            main_scope:
        """
        super(BuildPlaceholder, self).__init__(main_scope)

    def define_graph(self):
        """

        Returns:

        """
        PlaceholderConfig(
            builder=self,
            name='input',
            shape=[10, 10, 10],
            dtype=tf.int32
        )
        return [self.nx_graph.node['input']]

class TestPlaceholder(unittest.TestCase):

    """ Testing the placeholder wrapper. """

    def test_config_and_build(self):

        """ Testing the creation and building of a placeholder. """
        builder = BuildPlaceholder()
        builder.build_graph(builder.define_graph())
        builder.compute_graph.create_session()

        self.assertEqual(builder.nx_graph.number_of_nodes(), 1, msg='Incorrect number of nodes in the nx graph')
        self.assertEqual(builder.nx_graph.node['input']['config'].name, 'input')
        self.assertEqual(builder.nx_graph.node['input']['config'].shape, [10, 10, 10])
        self.assertEqual(builder.nx_graph.node['input']['config'].dtype, tf.int32)

        # the name of the node is the name of the config
        self.assertTrue('config' in builder.nx_graph.node['input'].keys(), msg='The config was not added to the graph')

        self.assertTrue('component_builder' in builder.nx_graph.node['input'].keys(),
                        msg='The component builder was not added to the '
                            'nx node.')

        self.assertTrue(builder.nx_graph.node['input']['is_build'], msg='Is build parameter was not updated')
        self.assertIsNotNone(builder.nx_graph.node['input']['output'], msg='The nx node output is None')


if __name__ == '__main__':

    unittest.main()
