""" Example"""

import tensorflow as tf
import numpy as np
from tf_graph_tool.components.dense.dense import DenseLayerConfig
from tf_graph_tool.components.recurrent.lstm import LSTMCellConfig
from tf_graph_tool.components.placeholder.placeholder import PlaceholderConfig
from tf_graph_tool.components.convolution.convolution_layer import Conv2DLayerConfig
from tf_graph_tool.builder.graph_builder import GraphBuilder
from tf_graph_tool.components.losses.a3c_loss import A3CLossConfig
from tf_graph_tool.components.trainer.rms_prob import RMSProbConfig


class ExampleNetwork(GraphBuilder):

    """ Example graph with recurrent cell. """

    def __init__(self, main_scope):
        """

        Initializes object

        Args:
            main_scope(str):

        """
        super(ExampleNetwork, self).__init__(main_scope)

    def define_graph(self):
        """

        Returns:
            build_nodes(list): List of nodes that mark the 'end' of the graph. Nodes in this list are unconnected from each other.

        """
        PlaceholderConfig(
            builder=self,
            name='input',
            shape=[None, 15]
        )
        PlaceholderConfig(
            builder=self,
            name='c_in',
            shape=[None, 10]
        )
        PlaceholderConfig(
            builder=self,
            name='h_in',
            shape=[None, 10]
        )
        PlaceholderConfig(
            builder=self,
            name='seq_len',
            shape=[],
            dtype=tf.int32
        )
        PlaceholderConfig(
            builder=self,
            name='learning_rate',
            shape=[]
        )
        PlaceholderConfig(
            builder=self,
            name='action_input',
            shape=[None, 4]
        )
        PlaceholderConfig(
            builder=self,
            name='target_value',
            shape=[None, 1]
        )
        PlaceholderConfig(
            builder=self,
            name='conv_in',
            shape=[None, 3, 3, 3]
        )

        DenseLayerConfig(
            builder=self,
            name='layer1',
            scope='encoder',
            num_units=8,
        )
        DenseLayerConfig(
            builder=self,
            name='layer2',
            scope='encoder',
            num_units=5,
            tensorboard_verbosity=1,
            type='hidden'
        )
        Conv2DLayerConfig(
            builder=self,
            name='conv_layer',
            scope='enc',
            type='hidden',
            filter=3,
            kernel_size=(3, 3),
            flatten_output=True  # flatten the output layer to rank one
        )
        LSTMCellConfig(
            builder=self,
            name='lstm1',
            scope='recurrent',
            binary=True,
            num_units=10,
            type='output',
            c_name='c_in',
            h_name='h_in',
            seq_len_name='seq_len'
        )

        DenseLayerConfig(
            builder=self,
            name='policy_layer',
            scope='policy',
            num_units=4,
            type='hidden'  # defines the dictionary for accessing the output op later.
        )
        DenseLayerConfig(
            builder=self,
            name='value_layer',
            scope='value',
            num_units=1,
            type='hidden'  # defines the dictionary for accessing the output op later.
        )

        A3CLossConfig(
            builder=self,
            name='loss',
            entropy_factor=0.01,
            logits_v_name='value_layer',
            logits_p_name='policy_layer',
            actions_name='action_input',
            target_v_name='target_value',
            tensorboard_verbosity=1,
            use_log_sigmoid=True,
        )

        RMSProbConfig(
            builder=self,
            name='rms_prob',
            scope='trainer',
            epsilon=1e-7,
            decay=0.99,
            use_grad_clip=True,
            learning_rate_name='learning_rate',
            momentum=0.0
        )
        self.add_edge('input', 'layer1')
        self.add_edge('layer1', 'layer2')
        self.add_edge('layer2', 'lstm1')
        self.add_edge('conv_in', 'conv_layer')
        self.add_edge('conv_layer', 'lstm1')
        self.add_edge('seq_len', 'lstm1')
        self.add_edge('c_in', 'lstm1')
        self.add_edge('h_in', 'lstm1')
        self.add_edge('lstm1', 'policy_layer')
        self.add_edge('lstm1', 'value_layer')
        self.add_edge('policy_layer', 'loss')
        self.add_edge('value_layer', 'loss')
        self.add_edge('target_value', 'loss')
        self.add_edge('action_input', 'loss')
        self.add_edge('loss', 'rms_prob')
        self.add_edge('learning_rate', 'rms_prob')

        return [self.nx_graph.nodes['rms_prob']]  # return as list


if __name__ == '__main__':

    builder = ExampleNetwork('my_graph')
    builder.build_graph(builder.define_graph())  # run the building mechanism

    graph = builder.compute_graph

    graph.initialize_graph_variables()
    policy = graph.session.run(
        graph.outputs['policy'],
        feed_dict={
            graph.inputs['input']: np.random.random_sample((1, 15)),
            graph.inputs['conv_in']: np.random.random_sample((1, 3, 3, 3)),
            graph.inputs['c_in']: np.zeros((1, 10)),
            graph.inputs['h_in']: np.zeros((1, 10))
        }
    )
    print "Policy: ", policy[0]
