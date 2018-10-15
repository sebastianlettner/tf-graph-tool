""" Testing the dense layer ."""

import unittest
import tensorflow as tf
import numpy as np
from tf_graph_tool.components.dense.dense import DenseLayerConfig
from tf_graph_tool.components.placeholder.placeholder import PlaceholderConfig
from tf_graph_tool.builder.graph_builder import GraphBuilder


class BuildDenseLayer(GraphBuilder):

    """ Class for testing """

    def __init__(self, main_scope='test_dense'):

        """

        Args:
            main_scope:
        """
        super(BuildDenseLayer, self).__init__(main_scope)

    def define_graph(self):

        DenseLayerConfig(
            builder=self,
            name='dense_layer',
            scope='outer_scope',
            num_units=10,
            type='hidden',
            activation=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.zeros_initializer,
            bias_initializer=tf.zeros_initializer,
            tensorboard_verbosity=2,
        )

        PlaceholderConfig(
            builder=self,
            name='input',
            shape=[None, 1]
        )
        self.add_edge('input', 'dense_layer')

        return [self.nx_graph.node['dense_layer']]


class TestDenseLayer(unittest.TestCase):

    """ Testing the dense layer wrapper. """

    def test_config_and_build(self):

        """ Testing the creation of the configuration.
            The following should happen:
                - the config is create with the specified parameters.
                - the config is added to the nx graph of the RootConfigClass

            Testing the building of the component.
            The following should happen:
                - The component appears in the tensorflow graph as specified
                - The variables of the component are recorded in the Variable manager of the neural network
        """

        """ Testing creation"""
        builder = BuildDenseLayer()
        self.assertEqual(builder.nx_graph.number_of_nodes(), 0, msg='The config graph is not empty')
        builder.build_graph(builder.define_graph())
        builder.compute_graph.create_session()

        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].name, 'dense_layer')
        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].scope, 'outer_scope')
        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].num_units, 10)
        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].type, 'hidden')
        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].activation, tf.nn.relu)
        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].use_bias, True)
        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].kernel_initializer, tf.zeros_initializer)
        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].bias_initializer, tf.zeros_initializer)
        self.assertEqual(builder.nx_graph.node['dense_layer']['config'].tensorboard_verbosity, 2)

        self.assertEqual(builder.nx_graph.number_of_nodes(), 2, msg='Incorrect number of nodes in the nx graph')

        # the name of the node is the name of the config
        self.assertTrue('config' in builder.nx_graph.node['dense_layer'].keys(),
                        msg='The config was not added to the graph')

        self.assertTrue('component_builder' in builder.nx_graph.node['dense_layer'].keys(),
                        msg='The component builder was not added to the '
                            'nx node.')

        with builder.compute_graph.tf_graph.as_default():
            self.assertEqual(len(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='test_dense')), 3,
                             msg='Tensorboard summaries missing')

        self.assertTrue(builder.nx_graph.node['input']['is_build'], msg='Is build parameter was not updated')
        self.assertTrue(builder.nx_graph.node['dense_layer']['is_build'], msg='Is build parameter was not updated')
        self.assertIsNotNone(builder.nx_graph.node['input']['output'], msg='The nx node output is None')
        self.assertIsNotNone(builder.nx_graph.node['dense_layer']['output'], msg='The nx node output is None')

        # get the var form the tf graphs
        with builder.compute_graph.tf_graph.as_default():
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='test_dense')

        self.assertEqual(len(train_vars), 2, msg="Incorrect number of variables in the tf graph")
        weights = train_vars[0]
        bias = train_vars[1]
        self.assertEqual(weights.name, 'test_dense/outer_scope/dense_layer/dense/kernel:0')
        self.assertEqual(bias.name, 'test_dense/outer_scope/dense_layer/dense/bias:0')

        builder.compute_graph.initialize_graph_variables()
        result = builder.compute_graph.session.run(builder.compute_graph.hidden['dense_layer'],
                                                   feed_dict={builder.compute_graph.inputs['input']: np.ones((1, 1))})[0]
        self.assertEqual(result.tolist(), [0]*10, msg='Wrong output')


if __name__ == '__main__':

    unittest.main()
