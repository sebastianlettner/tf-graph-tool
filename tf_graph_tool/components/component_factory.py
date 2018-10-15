""" Factory for components. """

from tf_graph_tool.components import *


class ComponentFactory(object):

    """ This factory produces instances of graph components. """

    @classmethod
    def produce(cls, config, inputs=list()):

        """

        Args:
            config(): Configuration of the component.
            inputs(list): Inputs for the component.

        Returns:

        """

        if config.factory_request == 'placeholder':
            return placeholder.placeholder.Placeholder(
                config=config
            )
        elif config.factory_request == 'dense':
            return dense.dense.DenseLayer(
                config=config,
                inputs=inputs
            )
        elif config.factory_request == 'lstm':
            return recurrent.lstm.LSTMCell(
                config=config,
                inputs=inputs
            )
        elif config.factory_request == 'conv_2D':
            return convolution.convolution_layer.Conv2DLayer(
                config=config,
                inputs=inputs
            )
        elif config.factory_request == 'a3c_loss':
            return losses.a3c_loss.A3CLoss(
                config=config,
                inputs=inputs
            )
        elif config.factory_request == 'classification_loss':
            return losses.classification.ClassificationLoss(
                config=config,
                inputs=inputs
            )
        elif config.factory_request == 'regression_loss':
            return losses.regression.RegressionLoss(
                config=config,
                inputs=inputs
            )
        elif config.factory_request == 'rms_prob':
            return trainer.rms_prob.RMSProb(
                config=config,
                inputs=inputs
            )
        else:
            raise ValueError('Unknown request for the component factory: {}'.format(config.factory_request))
