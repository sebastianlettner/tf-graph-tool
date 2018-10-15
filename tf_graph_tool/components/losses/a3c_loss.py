""" Policy Losses """

from tf_graph_tool.components.base_config import RootConfig
from tf_graph_tool.components.base_component import BaseComponent
from tf_graph_tool.util.tf_graph_utils import *
from tf_graph_tool.components.component_literals import LOSSES


class A3CLossConfig(RootConfig):

    """ Configuration for the policy loss. """

    def __init__(self,
                 builder,
                 name,
                 entropy_factor,
                 logits_v_name,
                 logits_p_name,
                 actions_name,
                 target_v_name,
                 tensorboard_verbosity=1,
                 use_log_sigmoid=True,
                 log_epsilon=1e-6
                 ):

        """

        Args:
            builder(GraphBuilder): The builder.
            name(str): Name of the component.
            entropy_factor(float): Factory for the entropy regualrizer. Recommended: [1e-4, 1e-3]
            logits_v_name(str): The name of the component that implements the logits for the value function.
            logits_p_name(str): The name of the component that implements the logits for the policy function.
            actions_name(str): The name of the component that implements the selected actions (Placeholder)
            target_v_name(str): The name of the component that implements the targeted value (Placeholder)
            tensorboard_verbosity(int):
                0: No tensorboard
                1: Tracking the output of the individual losses.
                2: 1, plus tracking of the activations of the outputs

            use_log_sigmoid(bool): Use the log sigmoid for the policy loss.
            log_epsilon(float): Small increment to avoid zeros in log
        """
        self._name = name
        self._entropy_phi = entropy_factor
        self._tensorflow_verbosity = tensorboard_verbosity
        self._use_log_sig = use_log_sigmoid
        self._log_epsilon = log_epsilon

        self._logits_v_name = logits_v_name
        self._logits_p_name = logits_p_name
        self._actions_name = actions_name
        self._target_v_name = target_v_name

        super(A3CLossConfig, self).__init__(
            builder=builder,
            factory_request='a3c_loss',
            type=LOSSES,
        )
        super(A3CLossConfig, self).add_node_to_nx_graph(self)

    @property
    def tensorboard_verbosity(self):
        return self._tensorflow_verbosity

    @property
    def entropy_phi(self):
        return self._entropy_phi

    @property
    def name(self):
        return self._name

    @property
    def use_log_sig(self):
        return self._use_log_sig

    @property
    def log_epsilon(self):
        return self.log_epsilon

    @property
    def logits_v_name(self):
        return self._logits_v_name

    @property
    def logits_p_name(self):
        return self._logits_p_name

    @property
    def actions_name(self):
        return self._actions_name

    @property
    def target_v_name(self):
        return self._target_v_name


class A3CLoss(BaseComponent):

    """
    The policy loss component expects logits as input that are not activated yet.
    The components will apply the activations.
    """
    def __init__(self,
                 config,
                 inputs):

        """
        Initializes object.
        Args:
            config(A3CLossConfig): The configuration
            inputs(list): The unactivated inputs.
        """
        self._inputs = inputs
        self._config = config

        self.softmax_p, self.value, self.cost_policy, self.cost_entropy, self.cost_value, total_cost,\
            self.selected_action, self.target_value = self.loss

        super(A3CLoss, self).__init__(
            name=self._config.name,
            output=total_cost,
            type=self._config.type,
            scope=None,
        )

    @graph_component
    def loss(self):
        """

        Returns:

        """
        logits_p = self.config.builder.compute_graph.get_component(self.config.logits_p_name)
        logits_v = self.config.builder.compute_graph.get_component(self.config.logits_v_name)
        selected_action = self.config.builder.compute_graph.get_component(self.config.actions_name)
        target_value = self.config.builder.compute_graph.get_component(self.config.target_v_name)

        if self._config.use_log_sig:
            softmax_p = tf.nn.softmax(logits_p)
            log_softmax_p = tf.nn.log_softmax(logits_p)
            log_selected_action_prob = tf.reduce_sum(log_softmax_p * selected_action, axis=1)

            cost_policy = log_selected_action_prob * (target_value - tf.stop_gradient(logits_v))
            cost_entropy = -1 * self.config.entropy_phi * \
                           tf.reduce_sum(log_softmax_p * softmax_p, axis=1)
            cost_policy = tf.reduce_sum(cost_policy, axis=1)
        else:
            softmax_p = tf.nn.softmax(logits_p)
            selected_action_prob = tf.reduce_sum(softmax_p * selected_action, axis=1)

            cost_policy = tf.log(tf.maximum(selected_action_prob, self.config.log_epsilon)) \
                          * (target_value - tf.stop_gradient(logits_v))
            cost_entropy = -1 * self.config.entropy_phi * \
                           tf.reduce_sum(tf.log(tf.maximum(softmax_p, self.config.log_epsilon)) *
                                         softmax_p, axis=1)
            cost_policy = tf.reduce_sum(cost_policy, axis=1)

        total_cost_policy = - (cost_policy + cost_entropy)

        cost_value = 0.5 * tf.reduce_sum(tf.square(target_value - logits_v), axis=1)

        total_cost = tf.reduce_sum(total_cost_policy + cost_value, axis=0)

        if self.config.tensorboard_verbosity == 0:
            return softmax_p, logits_v, cost_policy, cost_entropy, cost_value, total_cost, selected_action, target_value

        elif self.config.tensorboard_verbosity == 1:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.scalar('Entropy Loss', tf.squeeze(cost_entropy)))
            summaries.append(tf.summary.scalar('Policy Loss', tf.squeeze(cost_policy)))
            summaries.append(tf.summary.scalar('Value Loss', tf.squeeze(cost_value)))
            return softmax_p, logits_v, cost_policy, cost_entropy, cost_value, total_cost, selected_action, target_value

        elif self.config.tensorboard_verbosity == 2:

            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.scalar('Entropy Loss', tf.squeeze(cost_entropy)))
            summaries.append(tf.summary.scalar('Policy Loss', tf.squeeze(cost_policy)))
            summaries.append(tf.summary.scalar('Value Loss', tf.squeeze(cost_value)))
            summaries.append(tf.summary.histogram('Policy Activations', softmax_p))
            summaries.append(tf.summary.histogram('Value Activations', logits_v))
            return softmax_p, logits_v, cost_policy, cost_entropy, cost_value, total_cost, selected_action, target_value

        else:
            raise ValueError("Unsupported tensorboard verbosity {}. Supported are 0, 1, 2.".format(
                             self._config.tensorboard_verbosity))

    @property
    def config(self):
        return self._config

    def decorate_graph(self, compute_graph):

        """
        Add the output operation to the neural networks dictionary to make access easier.
        Also the policy activations will be added to the neural networks output dictionary with key: 'policy'
        Args:
            compute_graph(ComputeGraph):

        Returns:

        """
        compute_graph.inputs['selected_action'] = self.selected_action
        compute_graph.inputs['target_value'] = self.target_value
        compute_graph.outputs['policy'] = self.softmax_p
        compute_graph.outputs['value'] = self.value
        compute_graph.losses['value_loss'] = self.cost_value
        compute_graph.losses['policy_loss'] = self.cost_entropy
        compute_graph.losses['entropy_loss'] = self.cost_entropy
        compute_graph.losses['a3c_loss'] = self.output
