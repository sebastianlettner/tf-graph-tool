""" Testing the convolutional layer. """

import unittest
import tensorflow as tf
import numpy as np
from tf_graph_tool.components.convolution.convolution_layer import Conv2DLayerConfig
from tf_graph_tool.components.placeholder.placeholder import PlaceholderConfig
from tf_graph_tool.builder.graph_builder import GraphBuilder


class BuildConvLayer(GraphBuilder):

    """ Class for testing """

    def __init__(self, main_scope='test_conv'):

        """

        Args:
            main_scope:
        """
        super(BuildConvLayer, self).__init__(main_scope)

    def define_graph(self):

        """

        Returns:

        """
        Conv2DLayerConfig(
            builder=self,
            name='conv_layer',
            scope='outer_scope',
            type='hidden',
            filter=3,
            kernel_size=(3, 3),
            flatten_output=True,
            strides=(1, 1),
            padding='same',
            activations=tf.nn.relu,
            use_bias=True,
            kernel_initializer=tf.zeros_initializer,
            bias_initializer=tf.zeros_initializer,
            tensorboard_verbosity=2,
        )

        PlaceholderConfig(
            builder=self,
            name='input',
            shape=[None, 3, 3, 3]
        )
        self.add_edge('input', 'conv_layer')
        return [self.nx_graph.node['conv_layer']]


class TestConv2DLayer(unittest.TestCase):

    """ Testing the wrapper for convolutional. """

    def test_conv2d(self):

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
        builder = BuildConvLayer()
        self.assertEqual(builder.nx_graph.number_of_nodes(), 0, msg='The config graph is not empty')
        builder.build_graph(builder.define_graph())
        builder.compute_graph.create_session()

        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].name, 'conv_layer')
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].scope, 'outer_scope')
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].filter, 3)
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].type, 'hidden')
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].activations, tf.nn.relu)
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].use_bias, True)
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].flatten_output, True)
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].padding, 'same')
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].stride, (1, 1))
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].kernel_size, (3, 3))
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].kernel_initializer, tf.zeros_initializer)
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].bias_initializer, tf.zeros_initializer)
        self.assertEqual(builder.nx_graph.node['conv_layer']['config'].tensorboard_verbosity, 2)

        self.assertEqual(builder.nx_graph.number_of_nodes(), 2, msg='Incorrect number of nodes in the nx graph')
        self.assertTrue('config' in builder.nx_graph.node['conv_layer'].keys(),
                        msg='The config was not added to the graph')
        self.assertTrue('component_builder' in builder.nx_graph.node['conv_layer'].keys(),
                        msg='The component builder was not added to the nx node.')
        with builder.compute_graph.tf_graph.as_default():
            self.assertEqual(len(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='test_conv')), 3,
                             msg='Tensorboard summaries missing')
        self.assertTrue(builder.nx_graph.node['input']['is_build'], msg='Is build parameter was not updated')
        self.assertTrue(builder.nx_graph.node['conv_layer']['is_build'], msg='Is build parameter was not updated')
        self.assertIsNotNone(builder.nx_graph.node['input']['output'], msg='The nx node output is None')
        self.assertIsNotNone(builder.nx_graph.node['conv_layer']['output'], msg='The nx node output is None')

        # get the var form the tf graphs
        with builder.compute_graph.tf_graph.as_default():
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='test_conv')
        self.assertEqual(len(train_vars), 2, msg="Incorrect number of variables in the tf graph")
        weights = train_vars[0]
        bias = train_vars[1]
        self.assertEqual(weights.name, 'test_conv/outer_scope/conv_layer/conv2d/kernel:0')
        self.assertEqual(bias.name, 'test_conv/outer_scope/conv_layer/conv2d/bias:0')

        builder.compute_graph.initialize_graph_variables()
        result = builder.compute_graph.session.run(builder.compute_graph.hidden['conv_layer'],
                                                   feed_dict={builder.compute_graph.inputs['input']: np.ones((1, 3, 3, 3))})[0]
        self.assertEqual(result.tolist(), [0] * 27, msg='Wrong output')


if __name__ == '__main__':

    unittest.main()
