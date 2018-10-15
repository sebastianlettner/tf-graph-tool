""" Testing the recurrent component. """

import unittest
import tensorflow as tf
import numpy as np
from tf_graph_tool.components.recurrent.lstm import LSTMCellConfig
from tf_graph_tool.components.placeholder.placeholder import PlaceholderConfig
from tf_graph_tool.builder.graph_builder import GraphBuilder
from tf_graph_tool.util.tf_graph_utils import use_network_graph


class BuildRecurrentLayer(GraphBuilder):

    """ Class for testing """

    def __init__(self, main_scope='test_lstm'):

        """

        Args:
            main_scope:
        """
        super(BuildRecurrentLayer, self).__init__(main_scope)

    @use_network_graph
    def define_graph(self):
        """

        Returns:

        """
        LSTMCellConfig(
            builder=self,
            name='lstm_cell',
            scope='outer_scope',
            num_units=11,
            c_name='lstm_c_in',
            h_name='lstm_h_in',
            seq_len_name='seq_len',
            type='hidden',
            tensorboard_verbosity=1,
            binary=True,
            stochastic_method='bernoulli',
            preprocessing_method='sig',
            slope_tenor=tf.constant(0.99, name='slope_tensor')
        )
        PlaceholderConfig(
            builder=self,
            name='input',
            shape=[None, 3]
        )
        PlaceholderConfig(
            builder=self,
            name='lstm_c_in',
            shape=[None, 11]
        )
        PlaceholderConfig(
            builder=self,
            name='lstm_h_in',
            shape=[None, 11]
        )
        PlaceholderConfig(
            builder=self,
            name='seq_len',
            shape=[],
            dtype=tf.int32
        )
        self.add_edge('input', 'lstm_cell')
        self.add_edge('seq_len', 'lstm_cell')
        self.add_edge('lstm_c_in', 'lstm_cell')
        self.add_edge('lstm_h_in', 'lstm_cell')

        return [self.nx_graph.node['lstm_cell']]


class TestRecurrent(unittest.TestCase):

    """ Testing the recurrent module. """

    def test_lstm_create_and_build(self):

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
        builder = BuildRecurrentLayer()
        self.assertEqual(builder.nx_graph.number_of_nodes(), 0, msg='The config graph is not empty')

        builder.build_graph(builder.define_graph())
        builder.compute_graph.create_session()

        self.assertEqual(builder.nx_graph.node['lstm_cell']['config'].name, 'lstm_cell')
        self.assertEqual(builder.nx_graph.node['lstm_cell']['config'].scope, 'outer_scope')
        self.assertEqual(builder.nx_graph.node['lstm_cell']['config'].num_units, 11)
        self.assertEqual(builder.nx_graph.node['lstm_cell']['config'].type, 'hidden')
        self.assertEqual(builder.nx_graph.node['lstm_cell']['config'].binary, True)
        self.assertEqual(builder.nx_graph.node['lstm_cell']['config'].stochastic_method, 'bernoulli')
        self.assertEqual(builder.nx_graph.node['lstm_cell']['config'].preprocessing_method, 'sig')
        self.assertEqual(builder.nx_graph.node['lstm_cell']['config'].tensorboard_verbosity, 1)

        self.assertEqual(builder.nx_graph.number_of_nodes(), 5, msg='Incorrect number of nodes in the nx graph')
        self.assertTrue('config' in builder.nx_graph.node['lstm_cell'].keys(), msg='The config was not added to the graph')

        self.assertTrue('component_builder' in builder.nx_graph.node['lstm_cell'].keys(),
                        msg='The component builder was not added to the nx node.')
        with builder.compute_graph.tf_graph.as_default():
            self.assertEqual(len(tf.get_collection(tf.GraphKeys.SUMMARIES, scope='test_lstm')), 1,
                             msg='Tensorboard summaries missing')

        self.assertTrue(builder.nx_graph.node['input']['is_build'], msg='Is build parameter was not updated')
        self.assertTrue(builder.nx_graph.node['lstm_cell']['is_build'], msg='Is build parameter was not updated')
        self.assertIsNotNone(builder.nx_graph.node['input']['output'], msg='The nx node output is None')
        self.assertIsNotNone(builder.nx_graph.node['lstm_cell']['output'], msg='The nx node output is None')

        # get the var form the tf graphs
        with builder.compute_graph.tf_graph.as_default():
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='test_lstm')
        self.assertEqual(len(train_vars), 2, msg="Incorrect number of variables in the tf graph")
        weights = train_vars[0]
        bias = train_vars[1]
        self.assertEqual(weights.name, 'test_lstm/outer_scope/lstm_cell/binary_lstm/kernel:0')
        self.assertEqual(bias.name, 'test_lstm/outer_scope/lstm_cell/binary_lstm/bias:0')

        builder.compute_graph.initialize_graph_variables()
        _, _ = builder.compute_graph.session.run([
            builder.compute_graph.hidden['lstm_cell'],
            builder.compute_graph.hidden['lstm_cell_state_out']
            ],
            feed_dict={builder.compute_graph.inputs['input']: np.ones((1, 3)),
                       builder.compute_graph.inputs['lstm_c_in']: np.zeros((1, 11)),
                       builder.compute_graph.inputs['lstm_h_in']: np.zeros((1, 11)),
                       builder.compute_graph.inputs['seq_len']: 1
                       })

        _, state_out = builder.compute_graph.session.run([
            builder.compute_graph.hidden['lstm_cell'],
            builder.compute_graph.hidden['lstm_cell_state_out']
        ],
            feed_dict={builder.compute_graph.inputs['input']: np.ones((2, 3)),
                       builder.compute_graph.inputs['lstm_c_in']: np.zeros((2, 11)),
                       builder.compute_graph.inputs['lstm_h_in']: np.zeros((2, 11)),
                       builder.compute_graph.inputs['seq_len']: 1
                       })
        state_out


if __name__ == '__main__':

    unittest.main()
