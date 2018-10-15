""" Testing the trainer RMSProb. """

import unittest
import tensorflow as tf
import numpy as np
from tf_graph_tool.builder.graph_builder import GraphBuilder
from tf_graph_tool.util.tf_graph_utils import use_network_graph
from tf_graph_tool.components.placeholder.placeholder import PlaceholderConfig
from tf_graph_tool.components.dense.dense import DenseLayerConfig
from tf_graph_tool.components.losses.regression import RegressionLossConfig
from tf_graph_tool.components.trainer.rms_prob import RMSProbConfig


class RegerssionNetwork(GraphBuilder):

    """ For testing. """

    def __init__(self, main_scope='test_rms'):
        """

        Args:
            main_scope(str): The most outer variable scope.
        """

        super(RegerssionNetwork, self).__init__(main_scope)

    @use_network_graph
    def define_graph(self):
        """

        Returns:

        """

        PlaceholderConfig(
            builder=self,
            shape=[None, 1],
            name='input',
            tensorboard_verbosity=0
        )
        PlaceholderConfig(
            builder=self,
            shape=[None, 1],
            name='labels',
            tensorboard_verbosity=0
        )
        PlaceholderConfig(
            builder=self,
            shape=[],
            name='learning_rate',
            tensorboard_verbosity=0
        )
        DenseLayerConfig(
            builder=self,
            name='l1',
            scope='encoder',
            num_units=10,
            type='hidden',
            activation=tf.nn.relu
        )
        DenseLayerConfig(
            builder=self,
            name='l2',
            scope='encoder',
            num_units=10,
            type='hidden',
            activation=tf.nn.relu
        )
        DenseLayerConfig(
            builder=self,
            name='l3',
            scope='encoder',
            num_units=1,
            type='output'
        )
        RegressionLossConfig(
            builder=self,
            name='r_loss',
            tensorboard_verbosity=1,
            labels_name='labels',
            logits_name='l3'
        )
        RMSProbConfig(
            builder=self,
            name='rms_prob',
            scope='trainer',
            learning_rate_name='learning_rate',
            momentum=0.0,
            epsilon=1e-6,
            decay=0.99,
            use_grad_clip=False
        )
        self.add_edge('input', 'l1')
        self.add_edge('l1', 'l2')
        self.add_edge('l2', 'l3')
        self.add_edge('l3', 'r_loss')
        self.add_edge('labels', 'r_loss')
        self.add_edge('r_loss', 'rms_prob')
        self.add_edge('learning_rate', 'rms_prob')

        return [self.nx_graph.node['rms_prob']]


class TestRMSProb(unittest.TestCase):

    """ Testing the rms prob trainer by learning a quadratic function. """

    # output loss per 100 iteration
    debug = 0

    def test_rms_prob(self):

        """

        Returns:

        """
        builder = RegerssionNetwork()
        self.assertEqual(builder.nx_graph.number_of_nodes(), 0, msg='The root config graph is not empty')
        builder.build_graph(builder.define_graph())
        builder.compute_graph.create_session()

        self.assertEqual(builder.nx_graph.node['rms_prob']['config'].name, 'rms_prob')
        self.assertEqual(builder.nx_graph.node['rms_prob']['config'].type, 'output')
        self.assertEqual(builder.nx_graph.node['rms_prob']['config'].learning_rate_name, 'learning_rate')
        self.assertEqual(builder.nx_graph.node['rms_prob']['config'].scope, 'trainer')
        self.assertEqual(builder.nx_graph.node['rms_prob']['config'].momentum, 0.0)
        self.assertEqual(builder.nx_graph.node['rms_prob']['config'].epsilon, 1e-6)
        self.assertEqual(builder.nx_graph.node['rms_prob']['config'].decay, 0.99)
        self.assertEqual(builder.nx_graph.node['rms_prob']['config'].use_grad_clip, 0)

        self.assertEqual(builder.nx_graph.number_of_nodes(), 8, msg='Incorrect number of nodes in the nx graph')

        builder.compute_graph.initialize_graph_variables()
        xs = np.linspace(-3., 3., 600)
        ys = np.square(xs)

        for i in range(10000):
            builder.compute_graph.session.run(
                builder.compute_graph.outputs['rms_prob'],
                feed_dict={
                    builder.compute_graph.inputs['input']: np.reshape(xs, newshape=[600, 1]),
                    builder.compute_graph.inputs['learning_rate']: 0.001,
                    builder.compute_graph.inputs['labels']: np.reshape(ys, newshape=[600, 1])
                }
            )
            if TestRMSProb.debug:
                if i % 100 == 0:
                    print builder.compute_graph.session.run(
                        builder.compute_graph.losses['r_loss'],
                        feed_dict={
                            builder.compute_graph.inputs['input']: np.reshape(xs, newshape=[600, 1]),
                            builder.compute_graph.inputs['learning_rate']: 0.001,
                            builder.compute_graph.inputs['labels']: np.reshape(ys, newshape=[600, 1])
                        }
                    )


if __name__ == '__main__':

    unittest.main()