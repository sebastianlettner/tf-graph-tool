""" Regression Loss. """

from tf_graph_tool.components.base_config import RootConfig
from tf_graph_tool.components.base_component import BaseComponent
from tf_graph_tool.util.tf_graph_utils import *
from tf_graph_tool.components.component_literals import LOSSES


class RegressionLossConfig(RootConfig):

    """ Configuration of a regression loss """

    def __init__(self,
                 builder,
                 name,
                 labels_name,
                 logits_name,
                 tensorboard_verbosity=1):

        """

        Initializes object

        Args:
            builder(GraphBuilder): The builder.
            name(str): Name of this component
            labels_name(str): Name of the component that outputs the labels.
            logits_name(str): Name of the component that outputs the logits.
            tensorboard_verbosity(int):
                0: No tensorboard
                1: Tracking the output of the policy loss and the entropy loss.
        """

        self._name = name
        self._tensorflow_verbosity = tensorboard_verbosity
        self._labels_name = labels_name
        self._logits_name = logits_name

        super(RegressionLossConfig, self).__init__(
            builder=builder,
            factory_request='regression_loss',
            type=LOSSES,
        )
        super(RegressionLossConfig, self).add_node_to_nx_graph(self)

    @property
    def tensorboard_verbosity(self):
        return self._tensorflow_verbosity

    @property
    def name(self):
        return self._name

    @property
    def labels_name(self):
        return self._labels_name

    @property
    def logits_name(self):
        return self._logits_name


class RegressionLoss(BaseComponent):

    """ Implementation of a regression loss using MSE Error """

    def __init__(self,
                 config,
                 inputs):

        """
        Initializes object.
        Args:
            config(ClassificationLossConfig): The configuration
            inputs(list): The unactivated inputs.
        """
        self._inputs = inputs
        self._config = config

        loss = self.loss

        super(RegressionLoss, self).__init__(
            name=self._config.name,
            output=loss,
            type=self._config.type,
            scope=None,
        )

    @graph_component
    def loss(self):

        """

        Returns:

        """

        logits = self.config.builder.compute_graph.get_component(name=self.config.logits_name)
        labels = self.config.builder.compute_graph.get_component(name=self.config.labels_name)

        loss = tf.square(logits - labels)
        loss = tf.reduce_mean(loss, axis=0)

        if self.config.tensorboard_verbosity == 0:
            return loss

        elif self.config.tensorboard_verbosity == 1:
            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
            summaries.append(tf.summary.scalar(self.config.name, tf.squeeze(loss)))
            return loss

        else:
            raise ValueError("Unsupported tensorboard verbosity {}. Supported are 0, 1 for "
                             "classification losses.".format(self._config.tensorboard_verbosity))

    @property
    def config(self):
        return self._config

    def decorate_graph(self, compute_graph):

        """

        Args:
            compute_graph(ComputeGraph):

        Returns:

        """
        compute_graph.losses[self.config.name] = self.loss
