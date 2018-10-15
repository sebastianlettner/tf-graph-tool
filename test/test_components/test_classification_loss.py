""" Testing the policy component. """

import unittest
import tensorflow as tf
import numpy as np
from tf_graph_tool.components.losses.classification import ClassificationLossConfig
from tf_graph_tool.builder.graph_builder import GraphBuilder
from tf_graph_tool.util.tf_graph_utils import use_network_graph
from tf_graph_tool.components.placeholder.placeholder import PlaceholderConfig
from tf_graph_tool.components.dense.dense import DenseLayerConfig


class Classifier(GraphBuilder):

    """ Class for testing """

    def __init__(self, main_scope='test_policy_loss'):

        """

        Args:
            main_scope(str): Main scope of the graph
        """
        super(Classifier, self).__init__(main_scope)

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
            shape=[None, 2],
            name='labels',
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
            num_units=2
        )
        ClassificationLossConfig(
            builder=self,
            name='c_loss',
            tensorboard_verbosity=1,
            labels_name='labels',
            logits_name='l3'
        )
        self.add_edge('input', 'l1')
        self.add_edge('l1', 'l2')
        self.add_edge('l2', 'l3')
        self.add_edge('l3', 'c_loss')
        self.add_edge('labels', 'c_loss')
        return [self.nx_graph.node['c_loss']]


class TestA3CLoss(unittest.TestCase):

    """ Testing the a3c loss component. """

    def test_classification_loss(self):
        """ Testing the a3c loss """

        builder = Classifier()
        self.assertEqual(builder.nx_graph.number_of_nodes(), 0, msg='The root config graph is not empty')
        builder.build_graph(builder.define_graph())
        builder.compute_graph.create_session()

        self.assertEqual(builder.nx_graph.node['c_loss']['config'].name, 'c_loss')
        self.assertEqual(builder.nx_graph.node['c_loss']['config'].type, 'losses')
        self.assertEqual(builder.nx_graph.node['c_loss']['config'].tensorboard_verbosity, 1)
        self.assertEqual(builder.nx_graph.node['c_loss']['config'].labels_name, 'labels')
        self.assertEqual(builder.nx_graph.node['c_loss']['config'].logits_name, 'l3')

        self.assertEqual(builder.nx_graph.number_of_nodes(), 6, msg='Incorrect number of nodes in the nx graph')

        with builder.compute_graph.tf_graph.as_default():
            self.assertEqual(len(tf.get_collection(tf.GraphKeys.SUMMARIES)), 1,
                             msg='Tensorboard summaries missing')
        builder.compute_graph.initialize_graph_variables()
        _ = builder.compute_graph.session.run(
            builder.compute_graph.losses['c_loss'],
            feed_dict={
                builder.compute_graph.inputs['input']: np.random.random_sample((1, 10)),
                builder.compute_graph.inputs['labels']: np.random.random_sample((1, 2))
            }
        )


if __name__ == '__main__':

    unittest.main()
