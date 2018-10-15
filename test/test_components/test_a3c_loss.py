""" Testing the policy component. """

import unittest
import tensorflow as tf
import numpy as np
from tf_graph_tool.components.losses.a3c_loss import A3CLossConfig
from tf_graph_tool.builder.graph_builder import GraphBuilder
from tf_graph_tool.util.tf_graph_utils import use_network_graph
from tf_graph_tool.components.placeholder.placeholder import PlaceholderConfig
from tf_graph_tool.components.dense.dense import DenseLayerConfig
from tf_graph_tool.util.initializer import normalized_columns_initializer


class PolicyNetwork(GraphBuilder):

    """ Class for testing """

    def __init__(self, main_scope='test_policy_loss'):

        """

        Args:
            main_scope(str): Main scope of the graph
        """
        super(PolicyNetwork, self).__init__(main_scope)

    @use_network_graph
    def define_graph(self):

        """

        Returns:

        """
        PlaceholderConfig(
            builder=self,
            shape=[None, 10],
            name='input',
            tensorboard_verbosity=0
        )
        PlaceholderConfig(
            builder=self,
            shape=[None, 4],
            name='action_index',
            tensorboard_verbosity=0
        )
        PlaceholderConfig(
            builder=self,
            shape=[None, 1],
            name='target_v',
            tensorboard_verbosity=0
        )
        DenseLayerConfig(
            builder=self,
            name='share',
            scope='encoder',
            num_units=10,
            type='hidden',
        )
        DenseLayerConfig(
            builder=self,
            name='logits_p',
            scope='policy_layer',
            num_units=4,
            type='hidden',
            kernel_initializer=normalized_columns_initializer(),
            bias_initializer=tf.zeros_initializer
        )
        DenseLayerConfig(
            builder=self,
            name='logits_v',
            scope='value_layer',
            num_units=1
        )
        A3CLossConfig(
            builder=self,
            name='a3c_loss',
            entropy_factor=0.001,
            tensorboard_verbosity=1,
            logits_v_name='logits_v',
            logits_p_name='logits_p',
            actions_name='action_index',
            target_v_name='target_v'
        )
        self.add_edge('input', 'share')
        self.add_edge('share', 'logits_p')
        self.add_edge('share', 'logits_v')
        self.add_edge('logits_v', 'a3c_loss')
        self.add_edge('logits_p', 'a3c_loss')
        self.add_edge('action_index', 'a3c_loss')
        self.add_edge('target_v', 'a3c_loss')
        return [self.nx_graph.node['a3c_loss']]


class TestA3CLoss(unittest.TestCase):

    """ Testing the a3c loss component. """

    def test_policy_loss(self):
        """ Testing the a3c loss """

        builder = PolicyNetwork()
        self.assertEqual(builder.nx_graph.number_of_nodes(), 0, msg='The root config graph is not empty')
        builder.build_graph(builder.define_graph())
        builder.compute_graph.create_session()
        self.assertEqual(builder.nx_graph.node['a3c_loss']['config'].name, 'a3c_loss')
        self.assertEqual(builder.nx_graph.node['a3c_loss']['config'].type, 'losses')
        self.assertEqual(builder.nx_graph.node['a3c_loss']['config'].entropy_phi, 0.001)
        self.assertEqual(builder.nx_graph.node['a3c_loss']['config'].logits_v_name, 'logits_v')
        self.assertEqual(builder.nx_graph.node['a3c_loss']['config'].logits_p_name, 'logits_p')
        self.assertEqual(builder.nx_graph.node['a3c_loss']['config'].actions_name, 'action_index')
        self.assertEqual(builder.nx_graph.node['a3c_loss']['config'].target_v_name, 'target_v')
        self.assertEqual(builder.nx_graph.node['a3c_loss']['config'].tensorboard_verbosity, 1)

        self.assertEqual(builder.nx_graph.number_of_nodes(), 7, msg='Incorrect number of nodes in the nx graph')

        with builder.compute_graph.tf_graph.as_default():
            self.assertEqual(len(tf.get_collection(tf.GraphKeys.SUMMARIES)), 3,
                             msg='Tensorboard summaries missing')
        builder.compute_graph.initialize_graph_variables()
        policy, value = builder.compute_graph.session.run(
            [builder.compute_graph.outputs['policy'],
             builder.compute_graph.outputs['value']],
            feed_dict={
                builder.compute_graph.inputs['input']: np.random.random_sample((1, 10))
            }
        )
        policy = policy[0]
        for i in range(4):
            self.assertLessEqual(policy[i], 1)
            self.assertGreaterEqual(policy[i], 0)

        _ = builder.compute_graph.session.run(
            [builder.compute_graph.losses['policy_loss'],
             builder.compute_graph.losses['value_loss'],
             builder.compute_graph.losses['entropy_loss'],
             builder.compute_graph.losses['a3c_loss']],
            feed_dict={
                builder.compute_graph.inputs['input']: np.random.random_sample((1, 10)),
                builder.compute_graph.inputs['action_index']: np.asarray([0, 1, 0, 0]).reshape(1, 4),
                builder.compute_graph.inputs['target_v']: np.asarray([1]).reshape(1, 1)
            }
        )


if __name__ == '__main__':

    unittest.main()
