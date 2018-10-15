""" Implementation of rms prob trainer.  """

import tensorflow as tf
from tf_graph_tool.util.tf_graph_utils import *
from tf_graph_tool.components.base_config import RootConfig
from tf_graph_tool.components.base_component import BaseComponent
from tf_graph_tool.components.component_literals import OUTPUT


class RMSProbConfig(RootConfig):

    """ _configuration for RMSProb. """

    def __init__(self,
                 builder,
                 name,
                 scope,
                 learning_rate_name,
                 momentum,
                 epsilon,
                 decay=0.9,
                 use_grad_clip=True,
                 grad_clip_norm=40.0
                 ):

        """

        Initializes object.

        Args:
            builder(GraphBuilder): The builder
            name(str): Name for the trainer
            scope(str): Scope for the trainer.
            learning_rate_name(str): The name of the component that outputs the learning rate. E.g a placeholder or a constant
            momentum(float): Momentum parameter.
            epsilon(float): Epsilon parameter.
            decay(float): Discounting factor for the history/coming gradient
            use_grad_clip(bool): Clip the gradient by its norm
            grad_clip_norm(float): The gradient norm.
        """
        self._name = name
        self._learning_rate_name = learning_rate_name
        self._scope = scope
        self._momentum = momentum
        self._decay = decay
        self._epsilon = epsilon
        self._use_grad_clip = use_grad_clip
        self._grad_clip_norm = grad_clip_norm

        super(RMSProbConfig, self).__init__(
            builder=builder,
            factory_request='rms_prob',
            type=OUTPUT,
        )
        super(RMSProbConfig, self).add_node_to_nx_graph(self)

    @property
    def name(self):
        return self._name

    @property
    def learning_rate_name(self):
        return self._learning_rate_name

    @property
    def scope(self):
        return self._scope

    @property
    def momentum(self):
        return self._momentum

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def use_grad_clip(self):
        return self._use_grad_clip

    @property
    def grad_clip_norm(self):
        return self._grad_clip_norm

    @property
    def decay(self):
        return self._decay


class RMSProb(BaseComponent):

    """ Wrapper for tensorflow's RMSProb. The inputs to this components will be added together and then minimized. """

    def __init__(self,
                 config,
                 inputs):

        """
        Initializes object.

        Args:
            config(RMSProb_config): The _configuration.
            inputs(list): List of inputs. Should be loss operations and the learning rate parameter.
                          Losses will be added together.
        """
        self._inputs = inputs
        self._config = config

        self.global_step, train_op, self.learning_rate = self.trainer

        super(RMSProb, self).__init__(
            name=self._config.name,
            output=train_op,
            type=self._config.type,
            scope=self._config.scope,
        )

    @graph_component
    def trainer(self):
        """

        Returns:

        """
        with tf.variable_scope(self.config.scope):
            # remove the learning rate input from the losses
            learning_rate = self._config.builder.compute_graph.get_component(self._config.learning_rate_name)
            self._inputs.remove(learning_rate)

            total_loss = tf.reduce_sum(self._inputs, axis=0)

            global_step = tf.Variable(0, trainable=False, name='global_step')
            rms_prob = tf.train.RMSPropOptimizer(
                learning_rate=learning_rate,
                decay=self._config.decay,
                momentum=self._config.momentum,
                epsilon=self._config.epsilon,
                name=self._config.name
            )

            if self._config.use_grad_clip:
                grad_op = rms_prob.compute_gradients(total_loss)
                opt_grad_v_clipped = [(tf.clip_by_norm(g, self._config.grad_clip_norm), v)
                                           for g, v in grad_op if not g is None]
                train_op = rms_prob.apply_gradients(opt_grad_v_clipped, global_step=global_step)
            else:
                train_op = rms_prob.minimize(total_loss, global_step=global_step)

            return global_step, train_op, learning_rate

    @property
    def config(self):
        return self._config

    def decorate_graph(self, compute_graph):
        """

        Args:
            compute_graph(ComputeGraph):

        Returns:

        """
        compute_graph.inputs['learning_rate'] = self.learning_rate
        compute_graph.outputs[self._config.name] = self.output
        compute_graph.outputs['global_step'] = self.global_step
